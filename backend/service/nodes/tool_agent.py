"""工具 Agent 节点：规则优先 + LLM 兜底的工具调用与结果渲染（ADR-0004）。"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config.lm_config import lm_config
from backend.service.constants import USER_WAIT_LABEL
from backend.service.state import ServiceGraphState
from backend.service.tools import TOOL_MAP, TOOL_SPECS, ToolFailure, ToolNotFound, call_tool
from backend.utils.llm_utils import get_llm_client
from backend.utils.mongo_history_utils import get_recent_messages, save_chat_message
from backend.utils.sse_utils import push_progress

logger = logging.getLogger(__name__)

TOOL_SELECT_SYSTEM_PROMPT = """你是智服3D 的售后工具选择器，只输出 JSON，不要输出其他内容。
可选工具：
{specs}
规则：
1. 没有订单号时优先选 query_refund_policy；query_warranty 的 order_id 为必需参数，无法给出订单号时不要选它。
2. 工具名只能是上述列表之一，args 只包含工具声明的参数。
输出格式：{{"tool": "工具名", "args": {{...}}}}"""

TOOL_FAILURE_REPLY = "查询服务暂时不可用，已为您转接人工客服处理。"
ORDER_NOT_FOUND_REPLY = "未查询到订单 {order_id}，请核对订单号后重试。"
POLICY_NOT_FOUND_REPLY = "未找到该商品类型的退换货政策，请说明商品类别（如“3D打印机整机”）后重试。"
NEED_INFO_REPLY = "请补充订单号或商品类别，我才能帮您查询。"
WARRANTY_KEYWORDS = ("保修", "质保", "在保", "保内", "保外")


def _render_order(data: dict) -> str:
    return (
        f"已为您查到订单 {data.get('order_id')}：\n"
        f"- 商品：{data.get('product_name') or ''}（{data.get('product_model') or ''}）\n"
        f"- 金额：{data.get('amount')}\n"
        f"- 状态：{data.get('status')}\n"
        f"- 下单时间：{data.get('created_at')}"
    )


def _render_logistics(data: dict) -> str:
    lines = [f"订单 {data.get('order_id')} 的物流轨迹："]
    events = data.get("events") or []
    if not events:
        lines.append("- 暂无物流记录")
    for event in events:
        lines.append(f"- {event.get('event_time')} {event.get('node')}：{event.get('description')}")
    return "\n".join(lines)


def _render_warranty(data: dict) -> str:
    in_warranty = bool(data.get("in_warranty"))
    lines = [
        f"订单 {data.get('order_id')}（机型 {data.get('product_model')}）的保修情况：",
        f"- 保修周期：{data.get('warranty_start')} 至 {data.get('warranty_end')}（{data.get('warranty_months')} 个月）",
    ]
    if in_warranty:
        lines.append(f"- 当前状态：保修期内（剩余 {data.get('remaining_days')} 天）")
    else:
        lines.append("- 当前状态：已过保修期")
    if data.get("part"):
        part_status = "在保修范围内" if data.get("part_in_warranty") else "不在保修范围内（属于除外件）"
        lines.append(f"- 部件 {data.get('part')}：{part_status}")
    return "\n".join(lines)


def _render_policy(data: dict) -> str:
    excluded = data.get("excluded") or []
    return (
        f"{data.get('product_type')} 的退换货政策：\n"
        f"- 整机质保：{data.get('warranty_months')} 个月\n"
        f"- 无理由退货：{data.get('return_days')} 天\n"
        f"- 品质问题退换货：{data.get('exchange_days')} 天\n"
        f"- 除外件：{'、'.join(excluded) if excluded else '无'}\n"
        f"- 条款：{data.get('terms_text')}"
    )


RENDERERS = {
    "query_order": _render_order,
    "query_logistics": _render_logistics,
    "query_warranty": _render_warranty,
    "query_refund_policy": _render_policy,
}


class ToolAgent:
    """工具调用编排：命中订单号确定性串联工具，无订单号售后由 LLM 兜底选择。"""

    name: str = "node_tool_agent"

    def __call__(self, state: ServiceGraphState) -> ServiceGraphState:
        session_id = state.get("session_id", "")
        history = state.get("history") or get_recent_messages(session_id)
        state["history"] = history
        query = state.get("rewritten_query") or state.get("original_query", "")
        intent = state.get("intent", "consult")
        order_ids = state.get("order_ids") or []

        if state.get("is_stream"):
            push_progress(session_id, "querying", USER_WAIT_LABEL)
        if order_ids:
            results, failures = self._run_order_path(order_ids[0], intent, query)
        else:
            results, failures = self._run_llm_fallback(query)

        state["tool_results"] = results
        state["answer"] = self._assemble(results, failures)
        if failures:
            # 任一工具失败即代表“调用 + 重试”两次失败（F-TOOL-01），触发转人工。
            state["escalate_reason"] = "tool_failure"
            logger.info("工具调用失败，转人工: %s", state["tool_results"])

        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            intent=intent,
            models=state.get("models") or [],
        )
        return state

    def _run_order_path(self, order_id: str, intent: str, query: str) -> tuple[list[dict], int]:
        """订单号路径：问保修才走保修判定，否则串联订单 + 物流（修复退款/退货类误路由）。"""
        results: list[dict] = []
        failures = 0
        if intent == "after_sales" and any(keyword in query for keyword in WARRANTY_KEYWORDS):
            names = ["query_warranty"]
        else:
            names = ["query_order", "query_logistics"]
        for name in names:
            try:
                data = call_tool(name, {"order_id": order_id})
                results.append({"name": name, "ok": True, "data": data, "order_id": order_id})
            except ToolNotFound:
                results.append({"name": name, "ok": False, "not_found": True, "order_id": order_id})
            except ToolFailure as exc:
                logger.warning("工具 %s 失败: %s", name, exc)
                failures += 1
                results.append({"name": name, "ok": False, "error": True, "order_id": order_id})
        return results, failures

    def _run_llm_fallback(self, query: str) -> tuple[list[dict], int]:
        """无订单号的售后问题：LLM 在政策/保修之间选择工具与参数。"""
        selection = self._select_tool(query)
        if selection is None:
            return [{"name": "_need_info", "ok": False, "need_info": True}], 0
        name = selection["tool"]
        args = selection.get("args") or {}
        if name == "query_warranty" and not str(args.get("order_id") or "").strip():
            return [{"name": "_need_info", "ok": False, "need_info": True}], 0
        try:
            data = call_tool(name, args)
            return [{"name": name, "ok": True, "data": data}], 0
        except ToolNotFound:
            return [{"name": name, "ok": False, "not_found": True}], 0
        except ToolFailure as exc:
            logger.warning("工具 %s 失败: %s", name, exc)
            return [{"name": name, "ok": False, "error": True}], 1

    def _select_tool(self, query: str) -> dict | None:
        """LLM JSON 输出工具名与参数；解析失败返回 None 走补充信息话术。"""
        specs_text = "\n".join(
            f"- {spec['name']}: {spec['description']}；参数: {json.dumps(spec['params'], ensure_ascii=False)}"
            for spec in TOOL_SPECS
        )
        prompt = TOOL_SELECT_SYSTEM_PROMPT.format(specs=specs_text) + f"\n\n用户问题：{query}"
        try:
            llm = get_llm_client(model=lm_config.item_model, json_mode=True)
            content = str(
                llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)]).content or ""
            ).strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            name = str(result.get("tool") or "").strip()
            if name not in TOOL_MAP:
                raise ValueError(f"非法工具名: {name}")
            args = result.get("args") if isinstance(result.get("args"), dict) else {}
            return {"tool": name, "args": {str(k): v for k, v in args.items()}}
        except Exception as exc:
            logger.warning("工具选择失败，走补充信息话术: %s", exc)
            return None

    def _assemble(self, results: list[dict], failures: int) -> str:
        if failures:
            return TOOL_FAILURE_REPLY
        segments: list[str] = []
        for result in results:
            name = result["name"]
            if result.get("need_info"):
                return NEED_INFO_REPLY
            if result.get("not_found"):
                if name == "query_refund_policy":
                    return POLICY_NOT_FOUND_REPLY
                return ORDER_NOT_FOUND_REPLY.format(order_id=result.get("order_id") or "")
            if result.get("ok"):
                renderer = RENDERERS.get(name)
                if renderer:
                    segments.append(renderer(result["data"]))
        return "\n\n".join(segments) or "未查询到相关信息，请核对订单号或商品类别后重试。"
