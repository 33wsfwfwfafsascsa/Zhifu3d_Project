"""机型确认节点：提取机型 → kb_entities 对齐 → 确认 / 候选反问 / 兜底。"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config.lm_config import lm_config
from backend.config.milvus_config import milvus_config
from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.prompt.model_confirm import (
    MODEL_EXTRACT_SYSTEM_PROMPT,
    MODEL_EXTRACT_TEMPLATE,
)
from backend.processor.query_processor.state import QueryGraphState
from backend.utils.embedding_utils import generate_embeddings
from backend.utils.llm_utils import get_llm_client
from backend.utils.milvus_utils import get_milvus_client
from backend.utils.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_models,
)

logger = logging.getLogger(__name__)

CONFIRM_THRESHOLD = 0.85
CANDIDATE_THRESHOLD = 0.6
CANDIDATE_LIMIT = 3


class NodeModelConfirm(NodeBase[QueryGraphState]):
    """查询侧机型确认（ModelConfirmation，PRD F-INTENT-01）。"""

    name: str = "node_model_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id")
        original_query = state.get("original_query")
        if not session_id or not original_query:
            raise ValueError("session_id / original_query 不能为空")

        history = state.get("history") or get_recent_messages(session_id)
        state["history"] = history

        models, rewritten_query = self._extract_models(original_query, history)
        state["rewritten_query"] = rewritten_query

        align_result = self._align_models(models) if models else {"confirmed": [], "options": []}
        return self._apply_confirmation(state, align_result, history)

    def _extract_models(self, query: str, history: list[dict]) -> tuple[list[str], str]:
        """LLM 提取机型 + 指代消解 + 问题改写；失败回退原始问题。"""
        history_text = "\n".join(
            f"{msg.get('role', '')}: {msg.get('text', '')}" for msg in history if msg.get("text")
        )
        user_prompt = MODEL_EXTRACT_TEMPLATE.format(history_text=history_text, query=query)
        messages = [
            SystemMessage(content=MODEL_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = get_llm_client(model=lm_config.item_model, json_mode=True).invoke(messages)
            content = str(response.content or "").strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            models = [
                str(name).strip().replace(" ", "").replace("\n", "").replace("\t", "")
                for name in result.get("models", [])
            ]
            models = list(dict.fromkeys(name for name in models if name))
            rewritten_query = result.get("rewritten_query") or query
            return models, str(rewritten_query)
        except Exception as exc:
            logger.error("机型提取失败，回退原始问题: %s", exc)
            return [], query

    def _align_models(self, models: list[str]) -> dict[str, list[str]]:
        """逐机型做 kb_entities 混合检索，按阈值分确认 / 候选。"""
        client = get_milvus_client()
        if client is None:
            logger.error("Milvus 不可用，跳过机型对齐")
            return {"confirmed": [], "options": []}

        confirmed: list[str] = []
        options: list[str] = []
        for name in models:
            matches = self._search_entities(client, name)
            high = [match for match in matches if match["score"] > CONFIRM_THRESHOLD]
            if high:
                confirmed.extend(match["model"] for match in high)
                continue
            mid = [match for match in matches if match["score"] >= CANDIDATE_THRESHOLD]
            options.extend(match["model"] for match in mid[:CANDIDATE_LIMIT])
        return {
            "confirmed": list(dict.fromkeys(confirmed)),
            "options": list(dict.fromkeys(options)),
        }

    def _search_entities(self, client, name: str) -> list[dict]:
        """对单个机型名执行 kb_entities 稠密 COSINE 检索（BGE 归一化，分数 0-1）。"""
        try:
            embeddings = generate_embeddings([name])
            res = client.search(
                collection_name=milvus_config.entity_collection,
                data=[embeddings["dense"][0]],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE"},
                limit=5,
                output_fields=["model"],
            )
            if not res:
                return []
            return [
                {"model": hit.get("entity", {}).get("model"), "score": hit.get("distance")}
                for hit in res[0]
            ]
        except Exception as exc:
            logger.error("机型对齐检索失败: %s", exc)
            return []

    def _apply_confirmation(
        self,
        state: QueryGraphState,
        align_result: dict[str, list[str]],
        history: list[dict],
    ) -> QueryGraphState:
        """按确认状态更新 state，并回填历史消息的机型关联。"""
        confirmed = align_result["confirmed"]
        options = align_result["options"]

        if confirmed:
            state["models"] = confirmed
            state["needs_model_confirmation"] = False
            ids_to_update = [
                str(msg["_id"]) for msg in history if msg.get("_id") and not msg.get("models")
            ]
            if ids_to_update:
                update_message_models(ids_to_update, confirmed)
        elif options:
            state["models"] = []
            state["needs_model_confirmation"] = True
            state["answer"] = f"您是想问以下哪个机型：{'、'.join(options)}？请明确一下型号。"
        else:
            state["models"] = []
            state["needs_model_confirmation"] = True
            state["answer"] = "抱歉，未找到相关机型，请提供准确的打印机型号以便我为您查询。"

        user_message_id = state.get("user_message_id")
        if user_message_id:
            save_chat_message(
                session_id=state["session_id"],
                role="user",
                text=state["original_query"],
                rewritten_query=state.get("rewritten_query", ""),
                models=state.get("models") or [],
                message_id=user_message_id,
            )
        if state.get("answer"):
            save_chat_message(
                session_id=state["session_id"],
                role="assistant",
                text=state["answer"],
                models=state.get("models") or [],
            )
        return state
