"""RRF 融合节点：按倒数排名融合多路召回。"""

import logging

from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.state import QueryGraphState

logger = logging.getLogger(__name__)


class NodeRrf(NodeBase[QueryGraphState]):
    """对向量 / HyDE 两路召回做 RRF 融合排序。"""

    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # Milvus 原始命中结构为 {entity: {...}, distance: ...}，先解包 entity
        embedding_list = [
            doc.get("entity")
            for doc in (state.get("embedding_chunks") or [])
            if isinstance(doc, dict)
        ]
        hyde_list = [
            doc.get("entity")
            for doc in (state.get("hyde_embedding_chunks") or [])
            if isinstance(doc, dict)
        ]
        # 两路等权重融合
        merged = self._rrf_merge([(embedding_list, 1.0), (hyde_list, 1.0)])
        state["rrf_chunks"] = [doc for doc, _ in merged] # 只保留文档，丢分数
        return state

    def _rrf_merge(
        self,
        rrf_inputs: list[tuple[list[dict], float]],
        k: int = 60,
        max_results: int | None = None,
    ) -> list[tuple[dict, float]]:
        """RRF 公式融合：score += weight / (k + rank)。"""
        chunk_scores: dict = {} # chunk_id -> 累加 RRF 分
        chunk_data: dict = {} # chunk_id -> 第一条命中的完整 doc
        for rrf_input, weight in rrf_inputs:
            # rank 从 1 开始；排名越靠前（rank 越小），贡献越大
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get("chunk_id")
                if chunk_id is None:
                    continue # 无稳定主键的条目无法融合，跳过
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                chunk_data.setdefault(chunk_id, doc) # 保留第一次出现的数据
        # 按 RRF 分数倒序输出
        sorted_results = sorted(
            ((chunk_data[cid], score) for cid, score in chunk_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        return sorted_results[:max_results] if max_results else sorted_results
