"""查询流水线状态定义。"""

from typing import TypedDict


class QueryGraphState(TypedDict, total=False):
    session_id: str # 会话 ID（MongoDB 历史、SSE 通道共用）
    user_message_id: str # 用户消息的 Mongo _id（用于落库更新，而不是重复插入）
    original_query: str
    rewritten_query: str # LLM 改写后的独立完整问题（含机型/指代消解）
    models: list[str] # 已确认机型（检索过滤器；空 = 不做机型过滤）
    knowledge_types: list[str] # 知识类型过滤（如 manual / faq）
    history: list[dict] # 最近会话历史（Mongo 记录）
    embedding_chunks: list[dict]
    hyde_embedding_chunks: list[dict]
    web_search_docs: list[dict]
    rrf_chunks: list[dict]
    reranked_docs: list[dict]
    answer: str
    needs_model_confirmation: bool # 是否需要反问确认机型（当前策略始终 False）
    is_stream: bool # 是否流式输出（g 节点据此走 SSE delta）
