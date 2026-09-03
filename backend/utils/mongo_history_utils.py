"""MongoDB 会话历史读写：chat_message 集合，失败降级不阻断主链路。"""

import logging
from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient

from backend.config.mongo_config import mongo_config

logger = logging.getLogger(__name__)

_mongo_tool: "HistoryMongoTool | None" = None


class HistoryMongoTool:
    """封装 MongoDB 连接与 chat_message 集合。"""

    def __init__(self) -> None:
        if not mongo_config.url:
            raise RuntimeError("MONGO_URL 未配置，无法连接 MongoDB")
        self.client = MongoClient(mongo_config.url)
        self.db = self.client[mongo_config.db_name]
        self.chat_message = self.db["chat_message"]
        self.chat_message.create_index([("session_id", ASCENDING), ("ts", -1)])
        self.session = self.db["session"]
        self.session.create_index([("session_id", ASCENDING)], unique=True)
        self.session.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
        logger.info("MongoDB 已连接: %s", mongo_config.db_name)


def get_history_mongo_tool() -> "HistoryMongoTool | None":
    """获取 MongoDB 单例；连接失败返回 None 并降级为空历史。"""
    global _mongo_tool
    if _mongo_tool is None:
        try:
            _mongo_tool = HistoryMongoTool()
        except Exception as exc:
            logger.error("MongoDB 连接失败（历史读写降级）: %s", exc)
    return _mongo_tool


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    models: list[str] | None = None,
    image_urls: list[str] | None = None,
    intent: str | None = None,
    confidence: float | None = None,
    citations: list[dict] | None = None,
    escalate: bool | None = None,
    message_id: str | None = None,
) -> str:
    """写入/更新单条会话记录；message_id 存在时更新，否则新增。"""
    mongo_tool = get_history_mongo_tool()
    if mongo_tool is None:
        return ""

    document: dict[str, Any] = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "models": models or [],
        "image_urls": image_urls or [],
        "intent": intent,
        "confidence": confidence,
        "citations": citations or [],
        "escalate": escalate,
        "ts": datetime.now().timestamp(),
    }
    try:
        if message_id:
            mongo_tool.chat_message.update_one({"_id": ObjectId(message_id)}, {"$set": document})
            return message_id
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)
    except Exception as exc:
        logger.error("写入会话历史失败: %s", exc)
        return ""


def update_message_models(ids: list[str], models: list[str]) -> int:
    """批量回填历史消息的机型关联。"""
    mongo_tool = get_history_mongo_tool()
    if mongo_tool is None or not ids:
        return 0
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"models": models}},
        )
        return result.modified_count
    except Exception as exc:
        logger.error("更新历史机型关联失败: %s", exc)
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """查询指定会话最近 N 条记录（按时间正序）。"""
    mongo_tool = get_history_mongo_tool()
    if mongo_tool is None:
        return []
    try:
        cursor = mongo_tool.chat_message.find({"session_id": session_id}).sort("ts", ASCENDING).limit(limit)
        return list(cursor)
    except Exception as exc:
        logger.error("读取会话历史失败: %s", exc)
        return []


def clear_history(session_id: str) -> int:
    """清空指定会话历史。"""
    mongo_tool = get_history_mongo_tool()
    if mongo_tool is None:
        return 0
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        return result.deleted_count
    except Exception as exc:
        logger.error("清空会话历史失败: %s", exc)
        return 0
