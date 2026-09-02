"""客服编排常量：意图标签、knowledge_type 映射与规则兜底。"""

import re

INTENT_LABELS = ("consult", "troubleshoot", "after_sales", "complaint", "chitchat")

KNOWLEDGE_TYPE_BY_INTENT: dict[str, list[str]] = {
    "troubleshoot": ["troubleshooting", "faq", "manual"],
    "consult": ["manual", "faq"],
    "after_sales": ["policy"],
}

ORDER_ID_PATTERN = re.compile(r"\bORD-[A-Za-z0-9-]{4,}\b", re.IGNORECASE)

COMPLAINT_KEYWORDS = ("投诉", "举报", "差评", "非常不满意", "太差了", "我要退款")
AFTER_SALES_KEYWORDS = ("保修", "质保", "退换货", "退货", "换货", "退款政策")
CHITCHAT_KEYWORDS = ("你好", "您好", "在吗", "谢谢", "再见", "你是谁")


def rule_based_classify(text: str) -> tuple[str, float, list[str]]:
    """LLM 不可用时的规则兜底分类。"""
    if any(keyword in text for keyword in COMPLAINT_KEYWORDS):
        return "complaint", 0.95, []
    order_ids = list(dict.fromkeys(ORDER_ID_PATTERN.findall(text)))
    if order_ids:
        return "consult", 0.9, order_ids
    if any(keyword in text for keyword in AFTER_SALES_KEYWORDS):
        return "after_sales", 0.9, []
    stripped = text.strip()
    if len(stripped) <= 12 and any(keyword in stripped for keyword in CHITCHAT_KEYWORDS):
        return "chitchat", 0.9, []
    return "consult", 0.5, []
