"""Milvus 配置：从 .env 读取向量库参数。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MilvusConfig:
    milvus_url: str
    chunks_collection: str
    entity_collection: str


milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL", ""),
    chunks_collection=os.getenv("CHUNKS_COLLECTION", "kb_chunks"),
    entity_collection=os.getenv("ENTITY_COLLECTION", "kb_entities"),
)
