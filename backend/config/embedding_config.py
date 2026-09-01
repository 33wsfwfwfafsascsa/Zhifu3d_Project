"""Embedding 配置：本地 BGE-M3 模型参数。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_device: str
    bge_fp16: bool


embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH", ""),
    bge_device=os.getenv("BGE_DEVICE", "cpu"),
    bge_fp16=os.getenv("BGE_FP16", "0") in ("1", "True", "true", 1),
)
