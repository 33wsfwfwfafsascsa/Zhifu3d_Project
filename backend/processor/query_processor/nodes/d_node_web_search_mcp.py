"""联网搜索节点：默认关闭（ADR-0003），启用后经 DashScope MCP 检索公网。"""

import asyncio
import json
import logging

from backend.config.mcp_config import mcp_config
from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.state import QueryGraphState

logger = logging.getLogger(__name__)


class NodeWebSearchMcp(NodeBase[QueryGraphState]):
    """MCP 联网搜索，失败不影响主链路（PRD 7.2）。"""

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        if not mcp_config.enabled:
            logger.info("联网搜索已关闭（MCP_WEB_SEARCH_ENABLED=0），跳过")
            return {"web_search_docs": []}

        query = state.get("rewritten_query", "")
        if not query:
            return {"web_search_docs": []}
        try:
            result = asyncio.run(self._mcp_call(query))
            pages = json.loads(result.content[0].text).get("pages") or []
            docs: list[dict] = []
            for item in pages:
                snippet = (item.get("snippet") or "").strip()
                if not snippet:
                    continue
                docs.append(
                    {
                        "title": (item.get("title") or "").strip(),
                        "url": (item.get("url") or "").strip(),
                        "snippet": snippet,
                    }
                )
            return {"web_search_docs": docs}
        except Exception as exc:
            logger.error("联网搜索失败（不影响主链路）: %s", exc)
            return {"web_search_docs": []}

    async def _mcp_call(self, query: str):
        """调用 DashScope MCP 联网搜索工具。"""
        from agents.mcp import MCPServerStreamableHttp

        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.base_url,
                "Authorization": f"Bearer {mcp_config.api_key}",
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )
        try:
            await search_mcp.connect()
            return await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
        finally:
            await search_mcp.cleanup()
