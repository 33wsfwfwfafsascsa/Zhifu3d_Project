"""BGE-M3 混合检索节点：kb_chunks，带机型 / 知识类型过滤。"""

import logging

from backend.config.milvus_config import milvus_config
from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.state import QueryGraphState
from backend.utils.embedding_utils import generate_embeddings
from backend.utils.milvus_utils import (
    build_chunk_filter_expr, # 把机型/知识类型拼成 filter 表达式
    create_hybrid_search_requests, # 构建稠密/稀疏两个 AnnSearchRequest
    get_milvus_client,
    hybrid_search, # 执行混合检索 + WeightedRanker
)

logger = logging.getLogger(__name__)

# 检索结果需要返回的标量字段（Milvus output_fields）
OUTPUT_FIELDS = [
    "chunk_id",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "product_model",
    "knowledge_type",
]


class NodeSearchEmbedding(NodeBase[QueryGraphState]):
    """基于改写后问题 + 已确认机型执行混合检索。"""

    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 优先用改写后问题（包含机型/指代消解后的完整描述）
        query = state.get("rewritten_query") or state.get("original_query", "")
        client = get_milvus_client()
        if not query or client is None:
            return {"embedding_chunks": []} # 无客户端时不抛错，返回空召回

        # 构造过滤条件：models / knowledge_types 为空时返回 None（不过滤）
        expr = build_chunk_filter_expr(state.get("models"), state.get("knowledge_types"))
        try:
            embeddings = generate_embeddings([query])
            reqs = create_hybrid_search_requests(
                dense_vector=embeddings["dense"][0],
                sparse_vector=embeddings["sparse"][0],
                expr=expr,
                limit=10,
            )
            res = hybrid_search(
                client=client,
                collection_name=milvus_config.chunks_collection,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),# 稠密权重 0.8 > 稀疏 0.2
                limit=10,
                output_fields=OUTPUT_FIELDS,
            )
            # res[0] 是融合后的第一条结果列表（每项含 entity、distance）
            return {"embedding_chunks": res[0] if res else []}
        except Exception as exc:
            logger.error("混合检索失败: %s", exc)
            return {"embedding_chunks": []} # 检索失败 = 空召回，交给 RRF/兜底
