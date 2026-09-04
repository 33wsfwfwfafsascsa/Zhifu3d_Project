"""会话状态（Mongo session 集合）读写：转人工队列、原子领取、未解决计数（ADR-0005）。"""

import logging
from datetime import datetime
from typing import Any

from pymongo import ReturnDocument

from backend.utils.mongo_history_utils import get_history_mongo_tool

logger = logging.getLogger(__name__)

STATUS_ACTIVE = "active"
STATUS_ESCALATED = "escalated"
STATUS_PROCESSING = "processing"
STATUS_CLOSED = "closed"


def _now() -> float:
    return datetime.now().timestamp()


def get_session(session_id: str) -> dict[str, Any] | None:
    """读取会话状态文档；Mongo 不可用或不存在返回 None。"""
    tool = get_history_mongo_tool()
    if tool is None:
        return None
    try:
        return tool.session.find_one({"session_id": session_id})
    except Exception as exc:
        logger.error("读取会话状态失败: %s", exc)
        return None


def bump_streaks(session_id: str, negative: bool, not_found: bool) -> dict[str, Any]:
    """按本轮信号更新未解决计数：命中则 +1，未命中则清零（Q6 双口径）。"""
    tool = get_history_mongo_tool()
    if tool is None:
        return {}
    doc = get_session(session_id) or {}
    unresolved = (int(doc.get("unresolved_streak") or 0) + 1) if negative else 0
    not_found_streak = (int(doc.get("not_found_streak") or 0) + 1) if not_found else 0
    now = _now()
    try:
        tool.session.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "unresolved_streak": unresolved,
                    "not_found_streak": not_found_streak,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "session_id": session_id,
                    "status": STATUS_ACTIVE,
                    "created_at": now,
                },
            },
            upsert=True,
        )
    except Exception as exc:
        logger.error("更新未解决计数失败: %s", exc)
    return get_session(session_id) or {}


def mark_escalated(
    session_id: str,
    escalate_reason: str,
    summary: str,
    models: list[str],
    entities: dict[str, Any],
    citations: list[dict],
    ticket_draft: dict[str, Any],
) -> dict[str, Any] | None:
    """标记会话已转人工并写入载荷；状态已是 escalated/processing 时不动。"""
    tool = get_history_mongo_tool()
    if tool is None:
        return None
    now = _now()
    payload = {
        "$set": {
            "status": STATUS_ESCALATED,
            "escalate_reason": escalate_reason,
            "summary": summary,
            "models": models or [],
            "entities": entities or {},
            "citations": citations or [],
            "ticket_draft": ticket_draft,
            "updated_at": now,
        },
        "$setOnInsert": {"session_id": session_id, "created_at": now},
    }
    try:
        tool.session.update_one(
            {"session_id": session_id, "status": {"$nin": [STATUS_ESCALATED, STATUS_PROCESSING]}},
            payload,
            upsert=True,
        )
    except Exception as exc:
        logger.error("标记转人工失败: %s", exc)
    return get_session(session_id)


def list_queue() -> list[dict[str, Any]]:
    """坐席队列视图：status=escalated 且未认领的会话（按更新时间倒序）。"""
    tool = get_history_mongo_tool()
    if tool is None:
        return []
    try:
        return list(tool.session.find({"status": STATUS_ESCALATED}).sort("updated_at", -1))
    except Exception as exc:
        logger.error("查询转人工队列失败: %s", exc)
        return []


def list_handled(operator_name: str) -> list[dict[str, Any]]:
    """该坐席领取/接待过的会话（含会话 ID，按更新时间倒序）。"""
    tool = get_history_mongo_tool()
    if tool is None:
        return []
    try:
        return list(
            tool.session.find({"operator_name": operator_name}).sort("updated_at", -1)
        )
    except Exception as exc:
        logger.error("查询已接待会话失败: %s", exc)
        return []


def claim_session(session_id: str, operator_name: str) -> dict[str, Any] | None:
    """原子领取：仅 status=escalated 可被抢占为 processing（一人一单）。"""
    tool = get_history_mongo_tool()
    if tool is None:
        return None
    try:
        return tool.session.find_one_and_update(
            {"session_id": session_id, "status": STATUS_ESCALATED},
            {
                "$set": {
                    "status": STATUS_PROCESSING,
                    "operator_name": operator_name,
                    "updated_at": _now(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
    except Exception as exc:
        logger.error("领取会话失败: %s", exc)
        return None


def update_session(session_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    """按字段更新会话状态（工单回执、关闭等）。"""
    tool = get_history_mongo_tool()
    if tool is None:
        return None
    try:
        tool.session.update_one(
            {"session_id": session_id},
            {"$set": {**fields, "updated_at": _now()}},
        )
    except Exception as exc:
        logger.error("更新会话状态失败: %s", exc)
    return get_session(session_id)


def close_session(session_id: str) -> dict[str, Any] | None:
    """坐席关闭会话：processing → closed。"""
    return update_session(session_id, {"status": STATUS_CLOSED})
