"""查询流水线状态定义。"""

from typing import TypedDict


class QueryGraphState(TypedDict, total=False):
    session_id: str
    user_message_id: str
    original_query: str
    rewritten_query: str
    models: list[str]
    knowledge_types: list[str]
    history: list[dict]
    embedding_chunks: list[dict]
    hyde_embedding_chunks: list[dict]
    web_search_docs: list[dict]
    rrf_chunks: list[dict]
    reranked_docs: list[dict]
    answer: str
    needs_model_confirmation: bool
    is_stream: bool
