"""MinerU 配置：从 .env 读取 PDF 解析服务参数。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MinerUConfig:
    base_url: str
    api_token: str


mineru_config = MinerUConfig(
    base_url=os.getenv("MINERU_BASE_URL", ""),
    api_token=os.getenv("MINERU_API_TOKEN", ""),
)
