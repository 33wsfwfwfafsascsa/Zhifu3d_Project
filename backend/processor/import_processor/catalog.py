"""机型目录与知识类型推断：CLI 与导入 API 共用。"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__) # 本模块独立 logger


def load_model_catalog() -> list[str]:
    """从 MySQL products 读取去重后的标准机型目录；失败返回空列表。"""
    try:
        # 延迟导入：只有真正需要机型目录时才拉起 MySQL 依赖
        from backend.mock_business_api.config import get_connection

        conn = get_connection() # Mock 阶段每次新建连接
        try:
            with conn.cursor() as cur:
                # DISTINCT + ORDER BY：得到稳定、去重的权威目录
                cur.execute("SELECT DISTINCT model FROM products ORDER BY model")
                rows = cur.fetchall()
        finally:
            conn.close()
        # cursorclass=DictCursor，因此 row["model"] 可用
        return [row["model"] for row in rows]
    except Exception as exc:
        # 目录读取失败不阻断导入：只是“标注不做归一化”（e 节点回退 general）
        logger.warning("读取机型目录失败（%s），标注将不做归一化", exc)
        return []


def infer_knowledge_type(path: Path) -> str:
    """按 data/raw 目录名推断知识类型。
    目录名 -> 知识类型 的映射：
    manuals -> manual、faq -> faq、policy -> policy、troubleshooting -> troubleshooting
    推断不到返回空字符串，由调用方决定是否报错。
    """
    mapping = {
        "manuals": "manual",
        "faq": "faq",
        "policy": "policy",
        "troubleshooting": "troubleshooting",
    }
    return mapping.get(path.parent.name.lower(), "")
