"""意图分类节点：LLM 4+1 分类 + 规则兜底 + 订单号抽取。"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config.lm_config import lm_config
from backend.service.constants import (
    CHITCHAT_KEYWORDS,
    INTENT_LABELS,
    ORDER_ID_PATTERN,
    USER_WAIT_LABEL,
    rule_based_classify,
)
from backend.service.prompt.intent import INTENT_SYSTEM_PROMPT, INTENT_USER_TEMPLATE
from backend.service.state import ServiceGraphState
from backend.utils.llm_utils import get_llm_client
from backend.utils.sse_utils import push_progress

logger = logging.getLogger(__name__)


class IntentClassifier:
    """意图分类节点（ITEM_MODEL + JSON 输出）。"""

    name: str = "node_intent_classifier"

    def __call__(self, state: ServiceGraphState) -> ServiceGraphState:
        query = state.get("original_query", "")
        if state.get("is_stream"):
            push_progress(state.get("session_id"), "understanding", USER_WAIT_LABEL)
        intent, confidence, order_ids = self._classify(query)
        state["intent"] = intent
        state["confidence"] = confidence
        state["order_ids"] = order_ids
        logger.info("意图分类: intent=%s confidence=%.2f order_ids=%s", intent, confidence, order_ids)
        return state

    def _classify(self, query: str) -> tuple[str, float, list[str]]:
        try:
            llm = get_llm_client(model=lm_config.item_model, json_mode=True)
            prompt = INTENT_USER_TEMPLATE.format(query=query)
            content = str(
                llm.invoke(
                    [SystemMessage(content=INTENT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
                ).content
                or ""
            ).strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            intent = str(result.get("intent") or "").strip()
            if intent not in INTENT_LABELS:
                raise ValueError(f"非法意图标签: {intent}")
            confidence = min(max(float(result.get("confidence", 0.0)), 0.0), 1.0)
            order_ids = [str(item).strip() for item in result.get("order_ids", []) if str(item).strip()]
            # 闲聊校验：仅当消息是短问候/感谢等纯寒暄时接受 chitchat，避免业务问题被误判
            if intent == "chitchat" and not self._looks_like_chitchat(query):
                logger.info("LLM 误判闲聊，回退 consult: %s", query)
                intent, confidence = "consult", 0.5
        except Exception as exc:
            logger.warning("LLM 意图分类失败，启用规则兜底: %s", exc)
            intent, confidence, order_ids = rule_based_classify(query)

        # 规则补充：订单号正则兜底，与 LLM 结果取并集
        order_ids = list(dict.fromkeys(order_ids + ORDER_ID_PATTERN.findall(query)))
        return intent, confidence, order_ids

    def _looks_like_chitchat(self, query: str) -> bool:
        """纯寒暄判定：短消息且命中问候/感谢类关键词。"""
        stripped = query.strip()
        return len(stripped) <= 12 and any(keyword in stripped for keyword in CHITCHAT_KEYWORDS)
