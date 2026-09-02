"""MCP 联网搜索配置：默认关闭，P2 可按需启用（ADR-0003）。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class McpConfig:
    base_url: str
    api_key: str
    enabled: bool


mcp_config = McpConfig(
    base_url=os.getenv("MCP_DASHSCOPE_BASE_URL", ""),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    enabled=os.getenv("MCP_WEB_SEARCH_ENABLED", "0") in ("1", "True", "true", 1),
)
