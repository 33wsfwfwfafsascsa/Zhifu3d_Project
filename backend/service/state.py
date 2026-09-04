"""客服编排状态定义。"""

from typing import TypedDict


class ServiceGraphState(TypedDict, total=False):
    session_id: str
    user_message_id: str
    original_query: str
    rewritten_query: str
    history: list[dict]
    intent: str
    confidence: float
    order_ids: list[str]
    models: list[str]
    model_options: list[str]
    needs_model_confirmation: bool
    answer: str
    citations: list[dict]
    tool_results: list[dict]
    escalate: bool
    escalate_reason: str
    relaxed: bool
    summary: str
    ticket_draft: dict
    entities: dict
    is_stream: bool
    enable_web_search: bool
