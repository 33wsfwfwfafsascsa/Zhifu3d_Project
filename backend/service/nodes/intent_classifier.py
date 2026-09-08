"""意图分类节点：LLM 4+1 分类 + 规则兜底 + 订单号抽取。"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config.lm_config import lm_config # item_model 分类模型
from backend.service.constants import (
    CHITCHAT_KEYWORDS, # 寒暄词
    INTENT_LABELS, # 合法意图集合
    ORDER_ID_PATTERN, # 订单号正则
    USER_WAIT_LABEL, # “正在思考中”
    rule_based_classify, # LLM 失败兜底
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
        """LangGraph 节点入口：读问题 → 分类 → 写 state。"""
        query = state.get("original_query", "")

        # 流式模式下先给用户一个“理解中”的阶段反馈
        if state.get("is_stream"):
            push_progress(state.get("session_id"), "understanding", USER_WAIT_LABEL)
        intent, confidence, order_ids = self._classify(query)
        # 写入状态：这三个字段决定主图后续走哪条分支
        state["intent"] = intent
        state["confidence"] = confidence
        state["order_ids"] = order_ids
        logger.info("意图分类: intent=%s confidence=%.2f order_ids=%s", intent, confidence, order_ids)

        return state

    def _classify(self, query: str) -> tuple[str, float, list[str]]:
        """核心分类：LLM 优先、规则兜底、正则补单。"""
        try:
            llm = get_llm_client(model=lm_config.item_model, json_mode=True)
            prompt = INTENT_USER_TEMPLATE.format(query=query)
            content = str(
                llm.invoke(
                    [SystemMessage(content=INTENT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
                ).content
                or ""
            ).strip()
            # 兼容模型把 JSON 包在 ```json ... ``` 的情况
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            # --- 标签校验：非法标签直接当作失败处理（走规则兜底）---
            intent = str(result.get("intent") or "").strip()
            if intent not in INTENT_LABELS:
                raise ValueError(f"非法意图标签: {intent}")
            # --- 置信度夹取：防止模型输出 1.2 或 -0.1 ---
            confidence = min(max(float(result.get("confidence", 0.0)), 0.0), 1.0)
            # --- 订单号提取：清洗空白后保留非空项 ---
            order_ids = [str(item).strip() for item in result.get("order_ids", []) if str(item).strip()]

            # --- 闲聊防误判 ---
            # 模型容易把“你好，我的打印机一直堵头怎么办”判成 chitchat；
            # 这里要求整句是短寒暄才接受，否则强制回退 consult（0.5）
            if intent == "chitchat" and not self._looks_like_chitchat(query):
                logger.info("LLM 误判闲聊，回退 consult: %s", query)
                intent, confidence = "consult", 0.5
        except Exception as exc:
            # LLM 超时/JSON 解析失败/非法标签都会走这里
            logger.warning("LLM 意图分类失败，启用规则兜底: %s", exc)
            intent, confidence, order_ids = rule_based_classify(query)

        # --- 双保险：正则抽取订单号，与 LLM 结果并集去重 ---
        # 即使 LLM 漏提，只要文本里有 ORD-xxx 也能命中，从而走工具路径
        order_ids = list(dict.fromkeys(order_ids + ORDER_ID_PATTERN.findall(query)))
        return intent, confidence, order_ids

    def _looks_like_chitchat(self, query: str) -> bool:
        """纯寒暄判定：短消息且命中问候/感谢类关键词。"""
        stripped = query.strip()
        # 长度阈值 12 与 constants.CHITCHAT_KEYWORDS 配合使用
        return len(stripped) <= 12 and any(keyword in stripped for keyword in CHITCHAT_KEYWORDS)
