"""坐席工作台 API：会话队列 / 领取 / 上下文卡片 / 人工回复 / 工单提交（ADR-0005）。"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.service.tools import ToolFailure, create_ticket
from backend.utils.mongo_history_utils import get_recent_messages, save_chat_message
from backend.utils.session_utils import (
    STATUS_ESCALATED,
    STATUS_PROCESSING,
    claim_session,
    close_session,
    get_session,
    list_queue,
    update_session,
)
from backend.utils.sse_utils import SSEEvent, push_to_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


class TakeRequest(BaseModel):
    operator_name: str = Field(..., min_length=1)


class ReplyRequest(BaseModel):
    session_id: str
    operator_name: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class TicketSubmitRequest(BaseModel):
    session_id: str
    category: str | None = None
    summary: str | None = None
    customer_desc: str | None = None
    model: str | None = None


def _clean(doc: dict[str, Any] | None) -> dict[str, Any]:
    """剔除 Mongo 内部字段，返回可序列化文档。"""
    if not doc:
        return {}
    return {key: value for key, value in doc.items() if not key.startswith("_") and key != "_id"}


@router.get("/queue")
def queue() -> list[dict[str, Any]]:
    """转人工会话队列：status=escalated 且未认领。"""
    return [_clean(doc) for doc in list_queue()]


@router.post("/queue/{session_id}/take")
def take(session_id: str, request: TakeRequest) -> dict[str, Any]:
    """原子领取会话（一人一单）；已被领取返回 409。"""
    doc = claim_session(session_id, request.operator_name)
    if doc is None:
        current = get_session(session_id)
        if current and current.get("status") == STATUS_PROCESSING:
            raise HTTPException(status_code=409, detail="该会话已被其他坐席领取")
        raise HTTPException(status_code=404, detail="会话不存在或未处于待领取状态")
    return _clean(doc)


@router.get("/session/{session_id}")
def session_context(session_id: str) -> dict[str, Any]:
    """上下文卡片：会话状态 + 最近消息。"""
    doc = get_session(session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = [
        {
            "role": msg.get("role"),
            "text": msg.get("text"),
            "ts": msg.get("ts"),
        }
        for msg in get_recent_messages(session_id)
    ]
    return {"session": _clean(doc), "messages": messages}


@router.post("/reply")
def reply(request: ReplyRequest) -> dict[str, Any]:
    """坐席人工回复：写历史 + 推 operator 事件；仅处理中的会话可回复。"""
    doc = get_session(request.session_id)
    if doc is None or doc.get("status") != STATUS_PROCESSING:
        raise HTTPException(status_code=409, detail="会话未被领取或状态不允许回复")
    if doc.get("operator_name") != request.operator_name:
        raise HTTPException(status_code=403, detail="该会话已由其他坐席接待")
    message_id = save_chat_message(request.session_id, "operator", request.text)
    update_session(request.session_id, {"operator_last_reply_at": datetime.now().timestamp()})
    push_to_session(
        request.session_id,
        SSEEvent.OPERATOR,
        {
            "text": request.text,
            "operator_name": request.operator_name,
            "ts": datetime.now().timestamp(),
        },
    )
    return {"ok": True, "message_id": message_id}


@router.post("/tickets")
def submit_ticket(request: TicketSubmitRequest) -> dict[str, Any]:
    """按 AI 草稿（坐席可覆盖字段）创建工单，成功后推 ticket 回执。"""
    doc = get_session(request.session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    draft = doc.get("ticket_draft") or {}
    payload = {
        "category": request.category or draft.get("category", "咨询"),
        "summary": request.summary or draft.get("summary", ""),
        "customer_desc": request.customer_desc or draft.get("customer_desc", ""),
        "model": request.model or draft.get("model", ""),
    }
    try:
        ticket = create_ticket(
            session_id=request.session_id,
            category=payload["category"],
            summary=payload["summary"],
            customer_desc=payload["customer_desc"],
            model=payload["model"],
        )
    except ToolFailure as exc:
        logger.error("创建工单失败: %s", exc)
        raise HTTPException(status_code=502, detail=f"工单服务不可用: {exc}") from exc

    update_session(request.session_id, {"ticket_id": ticket.get("ticket_id")})
    push_to_session(
        request.session_id,
        SSEEvent.TICKET,
        {"ticket_id": ticket.get("ticket_id"), "status": ticket.get("status")},
    )
    return ticket


@router.post("/session/{session_id}/close")
def close(session_id: str) -> dict[str, Any]:
    """坐席关闭会话：processing → closed。"""
    doc = get_session(session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    closed = close_session(session_id)
    return _clean(closed)
