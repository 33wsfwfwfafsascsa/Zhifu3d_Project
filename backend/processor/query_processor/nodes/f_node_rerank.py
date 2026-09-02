"""Rerank 节点：DashScope 精排 + 断崖截断。"""

import logging

from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.state import QueryGraphState
from backend.utils.reranker_http_utils import rerank_documents

logger = logging.getLogger(__name__)

RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 3
RERANK_GAP_ABS: float = 0.5
RERANK_GAP_RATIO: float = 0.25


class NodeRerank(NodeBase[QueryGraphState]):
    """合并本地召回与联网结果，精排打分后按断崖截断。"""

    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        merged_docs = self._merge_multi_source_docs(state)
        ranked_docs = self._rerank_docs(state, merged_docs)
        state["reranked_docs"] = self._cliff_cutoff(ranked_docs)
        return state

    def _merge_multi_source_docs(self, state: QueryGraphState) -> list[dict]:
        docs: list[dict] = []
        for rrf_doc in state.get("rrf_chunks") or []:
            docs.append(
                {
                    "content": rrf_doc.get("content"),
                    "title": rrf_doc.get("title"),
                    "chunk_id": rrf_doc.get("chunk_id"),
                    "file_title": rrf_doc.get("file_title"),
                    "product_model": rrf_doc.get("product_model"),
                    "knowledge_type": rrf_doc.get("knowledge_type"),
                    "url": None,
                    "source": "local",
                }
            )
        for web_doc in state.get("web_search_docs") or []:
            docs.append(
                {
                    "content": web_doc.get("snippet"),
                    "title": web_doc.get("title"),
                    "chunk_id": None,
                    "file_title": None,
                    "product_model": None,
                    "knowledge_type": None,
                    "url": web_doc.get("url"),
                    "source": "web",
                }
            )
        return docs

    def _rerank_docs(self, state: QueryGraphState, merged_docs: list[dict]) -> list[dict]:
        if not merged_docs:
            return []
        query = state.get("rewritten_query") or state.get("original_query", "")
        try:
            scores = rerank_documents(query, [doc.get("content") or "" for doc in merged_docs])
            scored = [{**doc, "score": score} for doc, score in zip(merged_docs, scores)]
            return sorted(scored, key=lambda doc: doc["score"], reverse=True)
        except Exception as exc:
            logger.error("Rerank 失败，回退 RRF 顺序: %s", exc)
            return [{**doc, "score": None} for doc in merged_docs]

    def _cliff_cutoff(self, ranked_docs: list[dict]) -> list[dict]:
        """断崖截断：相邻分数差超过阈值时截断。"""
        if not ranked_docs:
            return []
        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)
        cutoff_pos = upper_bound
        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")
            if current_score is None or next_score is None:
                continue
            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)
            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                break
        return ranked_docs[:cutoff_pos]
