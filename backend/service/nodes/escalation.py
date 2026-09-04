"""转人工判定与输出节点：触发条件 + 摘要 + TicketDraft 落库（ADR-0005 / ADR-0008）。"""

import logging

from backend.service.nodes.rag_agent import NOT_FOUND_REPLY
from backend.service.state import ServiceGraphState
from backend.utils.llm_utils import get_llm_client
from backend.utils.mongo_history_utils import get_recent_messages, save_chat_message
from backend.utils.sse_utils import SSEEvent, push_to_session
from backend.utils.session_utils import (
    STATUS_ESCALATED,
    STATUS_PROCESSING,
    bump_streaks,
    get_session,
    mark_escalated,
)

logger = logging.getLogger(__name__)

NEGATIVE_KEYWORDS = (
    "还是不行",
    "依然",
    "还是那样",
    "没解决",
    "照样",
    "没效果",
    "还是有问题",
    "还是老样子",
)
EXPLICIT_KEYWORDS = ("转人工", "人工客服", "找客服", "真人客服", "客服电话", "联系客服")

CATEGORY_BY_INTENT = {
    "complaint": "投诉",
    "troubleshoot": "故障",
    "consult": "咨询",
    "after_sales": "售后",
}

ESCALATED_REPLY = "非常抱歉给您带来不便，已为您转接人工客服，请稍候。"
ALREADY_REPLY = "您的问题已转接人工客服处理中，请稍候。"

SUMMARY_PROMPT = """你是客服会话摘要员。请把下面的对话压缩成 2~3 句摘要，覆盖：用户的核心问题、涉及机型/订单号、已尝试的解决过程。只输出摘要文本，不要输出其他内容。"""


def escalation_check(state: ServiceGraphState) -> ServiceGraphState:
    """每轮结束前检查转人工触发条件。

    ADR-0008 起：空结果 + 低置信度不再即时转人工；保留投诉、显式要求、
    连续 2 轮未解决/空结果（工具失败由 tool_agent 置 escalate_reason）。
    """
    session_id = state.get("session_id", "")
    query = state.get("original_query", "")
    reason = state.get("escalate_reason") or ""

    if not reason and state.get("intent") == "complaint":
        reason = "complaint"
    if not reason and any(keyword in query for keyword in EXPLICIT_KEYWORDS):
        reason = "user_request"

    if not reason:
        negative = any(keyword in query for keyword in NEGATIVE_KEYWORDS)
        not_found = state.get("answer") == NOT_FOUND_REPLY
        doc = bump_streaks(session_id, negative, not_found)
        if doc.get("unresolved_streak", 0) >= 2 or doc.get("not_found_streak", 0) >= 2:
            reason = "unresolved"

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

    doc = get_session(session_id)
    if doc and doc.get("status") in (STATUS_ESCALATED, STATUS_PROCESSING):
        state["answer"] = ALREADY_REPLY
        state["escalate"] = True
        return state

    history = state.get("history") or get_recent_messages(session_id)
    summary = _generate_summary(history, state.get("original_query", ""))
    models = state.get("models") or []
    ticket_draft = {
        "category": CATEGORY_BY_INTENT.get(state.get("intent", "consult"), "咨询"),
        "summary": summary,
        "customer_desc": state.get("original_query", ""),
        "model": models[0] if models else "",
    }
    entities = {"order_ids": state.get("order_ids") or [], "ticket_ids": []}
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
    text = "\n".join(lines) + f"\nuser: {current_query}"
    try:
        response = get_llm_client().invoke(SUMMARY_PROMPT + "\n\n" + text[-6000:])
        summary = str(response.content or "").strip()
        return summary or current_query
    except Exception as exc:
        logger.warning("摘要生成失败，回退原话: %s", exc)
        return current_query
