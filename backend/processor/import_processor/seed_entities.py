"""kb_entities 种入脚本：以 MySQL products.model 为权威来源。"""

import logging
from typing import List

from backend.config.milvus_config import milvus_config
from backend.processor.import_processor.base import setup_logging
from backend.utils.embedding_utils import generate_embeddings
from backend.utils.milvus_utils import create_kb_entities_collection, get_milvus_client


def load_models_from_mysql() -> List[str]:
    """从 MySQL products 读取去重后的标准机型目录。"""
    from backend.mock_business_api.config import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT model FROM products ORDER BY model")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [row["model"] for row in rows]


def seed_entities(client, models: List[str]) -> int:
    """幂等种入：建表（如缺）→ 清空 → 向量化 → 批量插入。"""
    if not client:
        raise RuntimeError("Milvus 客户端不可用")
    if not models:
        raise RuntimeError("机型目录为空，请先运行 mock 业务 API 的 seed")

    collection = milvus_config.entity_collection # 目标集合名（默认 kb_entities）
    embeddings = generate_embeddings(models) # 机型名批量向量化（dense + sparse）
    vector_dim = len(embeddings["dense"][0]) # 从第一段向量推断维度

    # 集合不存在才创建并加载；已存在则复用（前提：schema/维度一致）
    if not client.has_collection(collection):
        logging.getLogger(__name__).info("创建集合 %s", collection)
        create_kb_entities_collection(client, collection, vector_dim)
        client.load_collection(collection)

    # 幂等清理：pk >= 0 删除全部现有记录（主键为 auto_id 的 INT64）
    client.delete(collection_name=collection, filter="pk >= 0")
    data = [
        {
            "model": model,
            "dense_vector": embeddings["dense"][i],
            "sparse_vector": embeddings["sparse"][i],
        }
        for i, model in enumerate(models)
    ]
    result = client.insert(collection_name=collection, data=data)
    return result.get("insert_count", 0)


def main() -> None:
    setup_logging()
    models = load_models_from_mysql()
    if not models:
        raise SystemExit("products 表为空，请先执行 mock 业务 API 的 seed.py")

    print(f"从 MySQL products 读取 {len(models)} 个机型：{models}")
    client = get_milvus_client()
    count = seed_entities(client, models)
    print(f"kb_entities 种入完成：{count} 条")


if __name__ == "__main__":
    main()
