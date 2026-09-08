"""Rerank 节点：DashScope 精排 + 断崖截断。"""

import logging

from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.state import QueryGraphState
from backend.utils.reranker_http_utils import rerank_documents

logger = logging.getLogger(__name__)

# 断崖截断参数
RERANK_MAX_TOPK: int = 10 # 精排后最多进入答案上下文 10 条
RERANK_MIN_TOPK: int = 3 # 无论如何至少保留 3 条（保护前 3 强相关）
RERANK_GAP_ABS: float = 0.5 # 断崖阈值（绝对，判断高分文档）
RERANK_GAP_RATIO: float = 0.25 # 断崖阈值（相对，判断低分文档）


class NodeRerank(NodeBase[QueryGraphState]):
    """合并本地召回与联网结果，精排打分后按断崖截断。"""

    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 本地（rrf）与公网（web）统一成同一结构
        merged_docs = self._merge_multi_source_docs(state)
        # 2. 精排
        ranked_docs = self._rerank_docs(state, merged_docs)
        # 3. 断崖截断后写状态
        state["reranked_docs"] = self._cliff_cutoff(ranked_docs)
        return state

    def _merge_multi_source_docs(self, state: QueryGraphState) -> list[dict]:
        """把不同来源的候选统一为同一 schema，方便后续答案/引用处理。"""
        docs: list[dict] = []
        # 本地知识库条目（e 节点已解包 entity，字段是扁平的）
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
        # 公网条目：snippet 充当 content，chunk 相关字段为空
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
        """调用精排模型打分并按分数倒序；失败保持合并顺序并打 None 分。"""
        if not merged_docs:
            return []
        query = state.get("rewritten_query") or state.get("original_query", "")
        try:
            # 返回与 merged_docs 顺序对齐的分数列表
            scores = rerank_documents(query, [doc.get("content") or "" for doc in merged_docs])
            scored = [{**doc, "score": score} for doc, score in zip(merged_docs, scores)]
            return sorted(scored, key=lambda doc: doc["score"], reverse=True) # 精排后重排
        except Exception as exc:
            # 精排服务故障：保留当前顺序（本地已是 RRF 序），不阻断问答
            logger.error("Rerank 失败，回退 RRF 顺序: %s", exc)
            return [{**doc, "score": None} for doc in merged_docs]

    def _cliff_cutoff(self, ranked_docs: list[dict]) -> list[dict]:
        """断崖截断：相邻分数差超过阈值时截断。"""
        if not ranked_docs:
            return []
        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs)) # 最多 10
        lower_bound = min(RERANK_MIN_TOPK, upper_bound) # 最少 3
        cutoff_pos = upper_bound # 默认全留
        # 从第 3 条开始与后一条比较（前 3 条受保护）
        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")
            if current_score is None or next_score is None:
                continue # 无分（降级模式）不触发截断
            # 计算相邻文档的分数绝对差距
            abs_gap = current_score - next_score
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）
            rel_gap = abs_gap / (abs(current_score) + 1e-6)
            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                cutoff_pos = idx + 1
                break
        return ranked_docs[:cutoff_pos]
