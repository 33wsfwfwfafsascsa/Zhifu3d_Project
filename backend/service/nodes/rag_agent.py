"""RAG Agent 节点：三路检索 → RRF → Rerank → 答案生成；机型不足时放宽 general。"""

import logging

from backend.processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
from backend.processor.query_processor.nodes.c_node_search_embedding_hyde import NodeSearchEmbeddingHyde
from backend.processor.query_processor.nodes.e_node_rrf import NodeRrf
from backend.processor.query_processor.nodes.f_node_rerank import NodeRerank
from backend.processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput
from backend.service.constants import KNOWLEDGE_TYPE_BY_INTENT
from backend.service.state import ServiceGraphState

logger = logging.getLogger(__name__)

NOT_FOUND_REPLY = "抱歉，没有找到与您问题相关的知识内容，建议转人工客服获取帮助。"
RELAXED_NOTE = "\n\n（注：未在您所提机型下找到相关内容，已放宽至通用知识。）"


class RagAgent:
    """知识检索 + 答案生成编排（单 Agent 最小闭环）。"""

    name: str = "node_rag_agent"

    def __init__(self) -> None:
        self.node_search = NodeSearchEmbedding()
        self.node_hyde = NodeSearchEmbeddingHyde()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer = NodeAnswerOutput()

    def __call__(self, state: ServiceGraphState) -> ServiceGraphState:
        intent = state.get("intent", "consult")
        knowledge_types = KNOWLEDGE_TYPE_BY_INTENT.get(intent, ["manual", "faq"])
        base_state: dict = {
            "session_id": state.get("session_id"),
            "original_query": state.get("original_query"),
            "rewritten_query": state.get("rewritten_query") or state.get("original_query"),
            "models": state.get("models") or [],
            "knowledge_types": knowledge_types,
            "history": state.get("history") or [],
        }

        reranked = self._retrieve(base_state)
        relaxed = False
        if not reranked and base_state["models"]:
            logger.info("机型严格过滤无结果，放宽至 general 检索")
            reranked = self._retrieve({**base_state, "models": ["general"]})
            relaxed = bool(reranked)

        if not reranked:
            state["answer"] = NOT_FOUND_REPLY
            state["citations"] = []
            state["relaxed"] = relaxed
            return state

        answer_state = {**base_state, "reranked_docs": reranked}
        result = self.node_answer(answer_state)
        answer = result.get("answer", "")
        if relaxed:
            answer += RELAXED_NOTE
        state["answer"] = answer
        state["citations"] = self._build_citations(reranked)
        state["relaxed"] = relaxed
        return state

    def _retrieve(self, base_state: dict) -> list[dict]:
        """向量 / HyDE 两路召回 → RRF → Rerank 断崖截断。"""
        try:
            embedding_result = self.node_search(dict(base_state))
            hyde_result = self.node_hyde(dict(base_state))
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
