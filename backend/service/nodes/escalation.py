"""转人工判定与输出节点：触发条件 + 摘要 + TicketDraft 落库。"""

import logging

# 复用 rag_agent 的“未找到”话术常量做相等判断
from backend.service.nodes.rag_agent import NOT_FOUND_REPLY
from backend.service.state import ServiceGraphState
from backend.utils.llm_utils import get_llm_client
from backend.utils.mongo_history_utils import get_recent_messages, save_chat_message
from backend.utils.sse_utils import SSEEvent, push_to_session
from backend.utils.session_utils import (
    STATUS_ESCALATED, # "escalated"
    STATUS_PROCESSING, # "processing"
    bump_streaks, # 更新连续未解决/空结果计数
    get_session, # 读会话状态
    mark_escalated, # 写转人工标记
)

logger = logging.getLogger(__name__)

# 用户说“还是不行/没解决/依然…”等负面反馈词
NEGATIVE_KEYWORDS = (
    "还是不行",
    "没用啊",
    "依然",
    "不行啊",
    "还是那样",
    "没解决",
    "解决不了",
    "照样",
    "没啥用",
    "没效果",
    "还是有问题",
    "还是老样子",
    "试了好多次",
    "还是报错",
    "毫无作用",
    "搞不定",
)
# 用户显式要求人工
EXPLICIT_KEYWORDS = (
    "转人工", 
    "人工客服", 
    "找客服", 
    "真人客服", 
    "客服电话", 
    "联系客服",
    "人工呢",
    "真人呢",
    "有活人吗",
    "呼叫客服",
    "听不懂人话",
    "不要机器人",
    "找经理",
    "在线客服",
)

CATEGORY_BY_INTENT = {
    "complaint": "投诉",
    "troubleshoot": "故障",
    "consult": "咨询",
    "after_sales": "售后",
}

ESCALATED_REPLY = "非常抱歉没能帮您解决问题，正在为您联系人工客服，请您稍等片刻。"
ALREADY_REPLY = "收到，您的需求已在优先处理中，人工客服正加速接入，请稍等一下哦。"

SUMMARY_PROMPT = """你是专业客服会话摘要员。请将输入的对话内容提炼为 2~3 句的接班摘要，供人工客服快速了解情况。

必须覆盖：
1. 用户核心问题与诉求；
2. 涉及的机型或订单号（若对话未提及，直接忽略，切勿捏造）；
3. 已尝试的排查步骤及当前结果。

极简输出要求：
仅输出摘要纯文本，严禁包含前导词（如“好的”、“摘要：”）、结束语或 Markdown 代码块。"""


def escalation_check(state: ServiceGraphState) -> ServiceGraphState:
    """每轮结束前检查转人工触发条件。
    空结果 + 低置信度不再即时转人工；保留投诉、显式要求、
    连续 2 轮未解决/空结果（工具失败由 tool_agent 置 escalate_reason）。
    """
    session_id = state.get("session_id", "")
    query = state.get("original_query", "")
    reason = state.get("escalate_reason") or "" # tool_agent 可能已写入 tool_failure

    # --- 触发源 1：投诉意图 ---
    if not reason and state.get("intent") == "complaint":
        reason = "complaint"

    # --- 触发源 2：显式要求人工 ---
    if not reason and any(keyword in query for keyword in EXPLICIT_KEYWORDS):
        reason = "user_request"

    # --- 触发源 3：连续两轮未解决 / 连续两轮空答案 ---
    if not reason:
        # negative：本轮问题文本里有没有“还是不行/没解决”等
        negative = any(keyword in query for keyword in NEGATIVE_KEYWORDS)
        # not_found：RAG 最终答案是否等于“抱歉无法回答”话术
        not_found = state.get("answer") == NOT_FOUND_REPLY
        # bump_streaks 内部做 upsert：命中 +1，未命中清零
        doc = bump_streaks(session_id, negative, not_found)
        # 双口径任意一条达到 2 就转人工
        if doc.get("unresolved_streak", 0) >= 2 or doc.get("not_found_streak", 0) >= 2:
            reason = "unresolved"
    # --- 命中任一触发源：写状态 ---
    if reason:
        state["escalate_reason"] = reason
        state["escalate"] = True
    return state


def escalation_output(state: ServiceGraphState) -> ServiceGraphState:
    """生成摘要与 TicketDraft 并落库；已转人工/处理中的会话幂等。"""
    reason = state.get("escalate_reason")
    if not reason:
        return state
    session_id = state.get("session_id", "")

    # --- 幂等保护：会话已经转人工/坐席处理中，不再重复创建 ---
    doc = get_session(session_id)
    if doc and doc.get("status") in (STATUS_ESCALATED, STATUS_PROCESSING):
        state["answer"] = ALREADY_REPLY
        state["escalate"] = True
        return state

    # --- 生成摘要与工单载荷 ---
    history = state.get("history") or get_recent_messages(session_id)
    summary = _generate_summary(history, state.get("original_query", ""))
    models = state.get("models") or []
    ticket_draft = {
        "category": CATEGORY_BY_INTENT.get(state.get("intent", "consult"), "咨询"),
        "summary": summary,
        "customer_desc": state.get("original_query", ""), # 用户原始问题作为描述
        "model": models[0] if models else "",  # 只取第一个机型
    }
    entities = {"order_ids": state.get("order_ids") or [], "ticket_ids": []}

    # mark_escalated 内部带状态过滤：escalated/processing 不会被覆盖
    mark_escalated(
        session_id,
        reason,
        summary,
        models,
        entities,
        state.get("citations") or [],
        ticket_draft,
    )

    state["summary"] = summary
    state["ticket_draft"] = ticket_draft
    state["entities"] = entities
    state["answer"] = ESCALATED_REPLY
    state["escalate"] = True

    # --- SSE 通知坐席队列/前端 ---
    push_to_session(
        session_id,
        SSEEvent.ESCALATE,
        {
            "summary": summary,
            "models": models,
            "citations": state.get("citations") or [],
            "ticket_draft": ticket_draft,
        },
    )
    save_chat_message(session_id, "assistant", ESCALATED_REPLY)
    logger.info("会话已转人工: session=%s reason=%s", session_id, reason)
    return state


def _generate_summary(history: list[dict], current_query: str) -> str:
    """LLM 压缩会话历史为 2~3 句摘要；失败回退用户原话。"""
    lines = [
        f"{msg.get('role', '')}: {msg.get('text', '')}"
        for msg in history
        if msg.get("text")
    ]
    # 当前用户问题追加在最后；只取末尾 6000 字符控制 token
    text = "\n".join(lines) + f"\nuser: {current_query}"
    try:
        response = get_llm_client().invoke(SUMMARY_PROMPT + "\n\n" + text[-6000:])
        summary = str(response.content or "").strip()
        return summary or current_query
    except Exception as exc:
        logger.warning("摘要生成失败，回退原话: %s", exc)
        return current_query
