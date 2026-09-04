"""RAG Agent 节点：三路检索 → RRF → Rerank → 答案生成；机型不足时放宽 general。"""

import logging
from concurrent.futures import ThreadPoolExecutor

from backend.config.mcp_config import mcp_config
from backend.processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
from backend.processor.query_processor.nodes.c_node_search_embedding_hyde import NodeSearchEmbeddingHyde
from backend.processor.query_processor.nodes.d_node_web_search_mcp import NodeWebSearchMcp
from backend.processor.query_processor.nodes.e_node_rrf import NodeRrf
from backend.processor.query_processor.nodes.f_node_rerank import NodeRerank
from backend.processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput
from backend.service.constants import KNOWLEDGE_TYPE_BY_INTENT, USER_WAIT_LABEL
from backend.service.state import ServiceGraphState
from backend.utils.sse_utils import push_progress
from backend.processor.query_processor.prompt.answer_prompt import NOT_FOUND_REPLY

logger = logging.getLogger(__name__)

RELAXED_NOTE = "\n\n（注：未在您所提机型下找到相关内容，已放宽至通用知识。）"
MODEL_HINT_NOTE = "\n\n（如需机型专属指引，请补充具体型号。）"
WEB_SEARCH_MIN_SCORE = 0.6


class RagAgent:
    """知识检索 + 答案生成编排（单 Agent 最小闭环）。"""

    name: str = "node_rag_agent"

    def __init__(self) -> None:
        self.node_search = NodeSearchEmbedding()
        self.node_hyde = NodeSearchEmbeddingHyde()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_web_search = NodeWebSearchMcp()
        self.node_answer = NodeAnswerOutput()

    def __call__(self, state: ServiceGraphState) -> ServiceGraphState:
        intent = state.get("intent", "consult")
        session_id = state.get("session_id") or ""
        is_stream = bool(state.get("is_stream"))
        knowledge_types = KNOWLEDGE_TYPE_BY_INTENT.get(intent, ["manual", "faq"])
        base_state: dict = {
            "session_id": state.get("session_id"),
            "original_query": state.get("original_query"),
            "rewritten_query": state.get("rewritten_query") or state.get("original_query"),
            "models": state.get("models") or [],
            "knowledge_types": knowledge_types,
            "history": state.get("history") or [],
            "is_stream": state.get("is_stream"),
        }

        if is_stream:
            push_progress(session_id, "searching", USER_WAIT_LABEL)
        reranked = self._retrieve(base_state)
        relaxed = False
        if not reranked and base_state["models"]:
            logger.info("机型严格过滤无结果，放宽至 general 检索")
            reranked = self._retrieve({**base_state, "models": ["general"]})
            relaxed = bool(reranked)

        # 联网兜底：本地无结果或精排最高分过低时触发（Day 5 决策：兜底触发 + 评测隔离）。
        if self._web_enabled(state):
            scores = [float(d["score"]) for d in reranked if d.get("score") is not None]
            if not reranked or not scores or max(scores) < WEB_SEARCH_MIN_SCORE:
                reranked = self._web_fallback(base_state, is_stream, session_id, reranked)

        if not reranked:
            state["answer"] = NOT_FOUND_REPLY
            state["citations"] = []
            state["relaxed"] = relaxed
            return state

        if is_stream:
            push_progress(session_id, "generating", USER_WAIT_LABEL)
        answer_state = {**base_state, "reranked_docs": reranked}
        result = self.node_answer(answer_state)
        answer = result.get("answer", "")
        if relaxed:
            answer += RELAXED_NOTE
        elif state.get("model_options") and not base_state["models"]:
            answer += MODEL_HINT_NOTE
        state["answer"] = answer
        state["citations"] = self._build_citations(reranked)
        state["relaxed"] = relaxed
        return state

    def _web_enabled(self, state: ServiceGraphState) -> bool:
        """联网兜底开关：请求显式开启 + MCP 启用 + 业务语境。

        离题/泛咨询（无机型且非故障/售后意图）不做联网兜底，避免用无关公网内容作答。
        """
        if not (bool(state.get("enable_web_search")) and mcp_config.enabled):
            return False
        if not state.get("models") and state.get("intent") not in ("troubleshoot", "after_sales"):
            return False
        return True
    def _web_fallback(
        self,
        base_state: dict,
        is_stream: bool,
        session_id: str,
        local_docs: list[dict],
    ) -> list[dict]:
        """本地不足时联网补充，并与本地结果合并精排；失败回退本地结果。"""
        if is_stream:
            push_progress(session_id, "searching_web", USER_WAIT_LABEL)
        try:
            web_result = self.node_web_search(dict(base_state))
        except Exception as exc:
            logger.error("联网搜索兜底失败: %s", exc)
            return local_docs
        web_docs = web_result.get("web_search_docs") or []
        if not web_docs:
            return local_docs
        merged_state: dict = {**base_state, "rrf_chunks": local_docs, "web_search_docs": web_docs}
        reranked_state = self.node_rerank(merged_state)
        return reranked_state.get("reranked_docs") or local_docs

    def _retrieve(self, base_state: dict) -> list[dict]:
        """向量 / HyDE 两路召回 → RRF → Rerank 断崖截断（Q6：机型严格过滤）。"""
        try:
            # 两路召回互不依赖，并发执行：HyDE 的大模型网络等待与向量检索重叠。
            with ThreadPoolExecutor(max_workers=2) as pool:
                embedding_future = pool.submit(self.node_search, dict(base_state))
                hyde_future = pool.submit(self.node_hyde, dict(base_state))
                embedding_result = embedding_future.result()
                hyde_result = hyde_future.result()
            query_state: dict = {**base_state, **embedding_result, **hyde_result, "web_search_docs": []}
            query_state = self.node_rrf(query_state)
            query_state = self.node_rerank(query_state)
            return query_state.get("reranked_docs") or []
        except Exception as exc:
            logger.error("检索链路异常: %s", exc, exc_info=True)
            return []

    def _build_citations(self, reranked_docs: list[dict]) -> list[dict]:
        """将 reranked_docs 映射为 Citation 结构（Q8 响应契约）。"""
        citations: list[dict] = []
        for doc in reranked_docs:
            citations.append(
                {
                    "source": doc.get("file_title") or doc.get("source") or "local",
                    "chunk_id": doc.get("chunk_id"),
                    "title": doc.get("title"),
                    "score": doc.get("score"),
                    "image_urls": self.node_answer._extract_images_from_docs([doc]),
                }
            )
        return citations
