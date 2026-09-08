"""业务工具：@tool 封装 + httpx 调 Mock 业务 API（8001）。
失败语义：
- ToolNotFound：404 属合法业务结果，友好兜底，不计工具失败；
- ToolFailure：连接错误/超时/5xx，重试 1 次（共 2 次尝试），连续 2 次失败转人工。
"""

import logging
from typing import Any

import httpx
from langchain_core.tools import tool # LangChain @tool：使普通函数可被 invoke

from backend.config.business_config import business_config

logger = logging.getLogger(__name__)

# 异常基类：便于上层统一 except ToolError 再细分
class ToolError(Exception):
    """业务工具调用基类异常。"""


class ToolNotFound(ToolError):
    """业务结果：资源不存在（404），不计工具失败。"""


class ToolFailure(ToolError):
    """基础设施失败：连接错误/超时/5xx，计失败次数。"""

# 总尝试次数 = 首次调用 + 1 次重试
MAX_ATTEMPTS = 2


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """带重试的 HTTP 调用；404 抛 ToolNotFound，5xx/网络失败重试后抛 ToolFailure。"""
    last_exc: Exception | None = None # 记录最后一次失败，循环结束后统一抛出
    for attempt in range(MAX_ATTEMPTS):
        try:
            with httpx.Client(timeout=business_config.timeout) as client:
                # 每次请求都新建 Client（简单场景够用；高并发可复用连接池）
                response = client.request(
                    method,
                    f"{business_config.base_url}{path}",
                    params=params,
                    json=json_body,
                )
        except httpx.HTTPError as exc:
            # 连接拒绝/超时/DNS 失败等：记录并进入下一次尝试
            last_exc = ToolFailure(f"业务接口连接失败: {exc}")
            logger.warning("工具调用失败（第 %d 次）: %s", attempt + 1, exc)
            continue

        # 404：合法业务结果（订单不存在/政策不存在），不重试
        if response.status_code == 404:
            raise ToolNotFound(f"{path} 资源不存在")

        # 5xx：服务端故障，值得重试一次
        if response.status_code >= 500:
            last_exc = ToolFailure(f"业务接口异常: HTTP {response.status_code}")
            logger.warning("工具调用失败（第 %d 次）: HTTP %d", attempt + 1, response.status_code)
            continue

        # 其他 4xx：参数/权限问题，重试无意义，直接失败
        if response.status_code >= 400:
            raise ToolFailure(f"业务接口拒绝: HTTP {response.status_code}")
        return response.json()
    raise last_exc or ToolFailure("业务接口不可用")

# ---------- 四个用户侧查询工具 ----------
@tool
def query_order(order_id: str) -> dict:
    """查询指定订单的状态、商品、金额与下单时间。参数 order_id 为订单号。"""
    return _request("GET", f"/api/orders/{order_id}")


@tool
def query_logistics(order_id: str) -> dict:
    """查询指定订单的物流轨迹列表。参数 order_id 为订单号。"""
    return _request("GET", f"/api/logistics/{order_id}")


@tool
def query_refund_policy(product_type: str) -> dict:
    """查询指定商品类型的退换货政策（保修月数/退货天数/换货天数/除外件）。参数 product_type 为商品类别。"""
    return _request("GET", "/api/refund-policy", params={"product_type": product_type})


@tool
def query_warranty(order_id: str, part: str | None = None) -> dict:
    """对指定订单做保修判定（保内/保外/剩余天数，可判定部件是否除外）。参数 order_id 为订单号，part 为可选部件名。"""
    params: dict[str, Any] = {}
    if part:
        params["part"] = part
    return _request("GET", f"/api/warranty/{order_id}", params=params)

# ---------- 内部工单创建（不进 TOOL_SPECS，不暴露给对话 LLM）----------
def create_ticket(
    session_id: str,
    category: str,
    summary: str,
    customer_desc: str = "",
    model: str = "",
) -> dict[str, Any]:
    """创建工单（仅转人工/坐席确认后内部调用，不对用户对话暴露）。"""
    return _request(
        "POST",
        "/api/tickets",
        json_body={
            "session_id": session_id,
            "category": category,
            "summary": summary,
            "customer_desc": customer_desc,
            "model": model,
        },
    )

# 完整工具注册表：规则路径（订单/物流/保修）与 call_tool 都从这里取
TOOL_MAP = {
    "query_order": query_order,
    "query_logistics": query_logistics,
    "query_refund_policy": query_refund_policy,
    "query_warranty": query_warranty,
}

# 仅供 LLM 兜底选择：无订单号的售后问题在政策/保修之间二选一。
TOOL_SPECS = [
    {
        "name": "query_refund_policy",
        "description": "查询指定商品类型的退换货政策（保修月数/退货天数/换货天数/除外件），适合没有订单号的退换货/保修政策类问题",
        "params": {"product_type": "商品类别，如 3D打印机整机"},
    },
    {
        "name": "query_warranty",
        "description": "对指定订单做保修判定（保内/保外/剩余天数），适合已有订单号的保修判定问题",
        "params": {"order_id": "订单号（必需）", "part": "部件名（可选，判定该部件是否在保）"},
    },
]


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """按名称分发调用用户侧工具；未知工具抛 ToolFailure。"""
    tool_obj = TOOL_MAP.get(name)
    if tool_obj is None:
        raise ToolFailure(f"未知工具: {name}")
    return tool_obj.invoke(dict(args))
