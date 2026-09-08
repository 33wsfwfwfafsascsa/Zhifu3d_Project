"""联网搜索节点：经百炼 DashScope WebSearch MCP 检索公网，兜底触发。"""

import json
import logging

import httpx # 异步优先的 HTTP 客户端（此处用同步 Client）

from backend.config.mcp_config import mcp_config
from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.state import QueryGraphState

logger = logging.getLogger(__name__)


class NodeWebSearchMcp(NodeBase[QueryGraphState]):
    """MCP 联网搜索，失败不影响主链路。百炼 MCP 为无状态 HTTP，直接 JSON-RPC 调用。"""

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 配置关闭：最快路径，连客户端都不建
        if not mcp_config.enabled:
            logger.info("联网搜索已关闭（MCP_WEB_SEARCH_ENABLED=0），跳过")
            return {"web_search_docs": []}

        query = state.get("rewritten_query", "") # 注意：这里没有回退 original_query
        if not query:
            return {"web_search_docs": []}
        try:
            pages = self._search_pages(query)
            docs: list[dict] = []
            for item in pages:
                snippet = (item.get("snippet") or "").strip()
                if not snippet:
                    continue # 没有摘要的公网条目对 RAG 无用，直接丢弃
                docs.append(
                    {
                        "title": (item.get("title") or "").strip(),
                        "url": (item.get("url") or "").strip(),
                        "snippet": snippet,
                    }
                )
            return {"web_search_docs": docs}
        except Exception as exc:
            # 联网兜底必须“尽力而为”：失败只告警
            logger.error("联网搜索失败（不影响主链路）: %s", exc)
            return {"web_search_docs": []}

    def _search_pages(self, query: str, count: int = 5) -> list[dict]:
        """初始化握手 + 调用 bailian_web_search，返回 pages 列表。"""
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if mcp_config.api_key:
            # MCP 接口鉴权（配置里默认复用 OPENAI_API_KEY）
            headers["Authorization"] = f"Bearer {mcp_config.api_key}"
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "zhifu3d", "version": "1.0"},
            },
        }
        call_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "bailian_web_search",
                "arguments": {"query": query, "count": count},
            },
        }
        # 两次 POST 到同一 base_url；无状态 MCP 无需维持会话
        with httpx.Client(timeout=15.0) as client:
            client.post(mcp_config.base_url, json=init_payload, headers=headers).raise_for_status()
            response = client.post(mcp_config.base_url, json=call_payload, headers=headers)
            response.raise_for_status()
        body = response.json()
        result = body.get("result") or {}
        content = result.get("content") or [] # MCP 工具结果 content 数组
        if not content or result.get("isError"):
            return []
        # content[0].text 是 JSON 字符串，里面是 pages
        data = json.loads(content[0].get("text") or "{}")
        return data.get("pages") or []
