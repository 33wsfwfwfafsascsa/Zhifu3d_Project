"""Rerank 配置：DashScope TextReRank 参数。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    api_key: str
    model: str
    instruct: bool


reranker_config = RerankerConfig(
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    model=os.getenv("TEXT_RERANK_MODEL", ""),
    instruct=os.getenv("TEXT_RERANK_INSTRUCT", "false") in ("1", "True", "true", 1),
)
