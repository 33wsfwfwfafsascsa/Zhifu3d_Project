"""机型目录与知识类型推断：CLI 与导入 API 共用。"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model_catalog() -> list[str]:
    """从 MySQL products 读取去重后的标准机型目录；失败返回空列表。"""
    try:
        from backend.mock_business_api.config import get_connection

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT model FROM products ORDER BY model")
                rows = cur.fetchall()
        finally:
            conn.close()
        return [row["model"] for row in rows]
    except Exception as exc:
        logger.warning("读取机型目录失败（%s），标注将不做归一化", exc)
        return []


def infer_knowledge_type(path: Path) -> str:
    """按 data/raw 目录名推断知识类型。"""
    mapping = {
        "manuals": "manual",
        "faq": "faq",
        "policy": "policy",
        "troubleshooting": "troubleshooting",
    }
    return mapping.get(path.parent.name.lower(), "")
