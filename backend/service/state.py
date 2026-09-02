"""客服编排状态定义。"""

from typing import TypedDict


class ServiceGraphState(TypedDict, total=False):
    session_id: str
    user_message_id: str
    original_query: str
    intent: str
    confidence: float
    order_ids: list[str]
    models: list[str]
    needs_model_confirmation: bool
    answer: str
    citations: list[dict]
    escalate: bool
    relaxed: bool
