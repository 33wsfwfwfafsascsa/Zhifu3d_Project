"""HyDE 检索节点：LLM 生成假设文档后用其向量召回，提升短问题召回率。"""

import logging

from backend.config.milvus_config import milvus_config
from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from backend.processor.query_processor.state import QueryGraphState
from backend.utils.embedding_utils import generate_embeddings
from backend.utils.llm_utils import get_llm_client
from backend.utils.milvus_utils import (
    build_chunk_filter_expr,
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)

logger = logging.getLogger(__name__)


class NodeSearchEmbeddingHyde(NodeBase[QueryGraphState]):
    """HyDE (Hypothetical Document Embedding) 检索节点。
       核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。
    """

    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query = state.get("rewritten_query") or state.get("original_query", "")
        if not rewritten_query:
            return {"hyde_embedding_chunks": []}
        try:
            # 1. LLM 生成假设文档
            hyde_doc = self._create_hyde_doc(rewritten_query)
            # 2. 用“问题 + 假设文档”向量检索
            return {
                "hyde_embedding_chunks": self._search_hyde(rewritten_query, hyde_doc, state),
            }
        except Exception as exc:
            # HyDE 整路失败只影响这一路召回，不影响另一路
            logger.error("HyDE 检索失败: %s", exc)
            return {"hyde_embedding_chunks": []}

    def _create_hyde_doc(self, rewritten_query: str) -> str:
        """
        利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
        HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
        从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。
        """
        llm = get_llm_client() # 默认主模型（非 item_model）
        prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
        return str(llm.invoke(prompt).content or "").strip()

    def _search_hyde(
        self,
        rewritten_query: str,
        hyde_doc: str,
        state: QueryGraphState,
    ) -> list[dict]:
        client = get_milvus_client()
        if client is None:
            return []
        # 假设文档与真实问题拼接，让向量更接近“文档侧语言”
        combined_text = f"{rewritten_query} {hyde_doc}"
        expr = build_chunk_filter_expr(state.get("models"), state.get("knowledge_types"))
        embeddings = generate_embeddings([combined_text])
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
            ranker_weights=(0.8, 0.2),
            limit=10,
            output_fields=[
                "chunk_id",
                "content",
                "title",
                "parent_title",
                "part",
                "file_title",
                "product_model",
                "knowledge_type",
            ],
        )
        return res[0] if res else []
