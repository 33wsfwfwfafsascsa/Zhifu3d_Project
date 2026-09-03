"""Mock 业务 API（8001）连接配置。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class BusinessConfig:
    base_url: str
    timeout: float


business_config = BusinessConfig(
    base_url=os.getenv("MOCK_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/"),
    timeout=float(os.getenv("BUSINESS_API_TIMEOUT", "10")),
)
