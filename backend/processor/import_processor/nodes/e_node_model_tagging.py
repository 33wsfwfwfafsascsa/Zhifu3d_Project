"""机型标注节点：文件级机型直写 / 多机型文件逐 chunk LLM 标注，归一化到权威目录。"""

import logging
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.config.lm_config import lm_config
from backend.processor.import_processor.config import KNOWLEDGE_TYPES, MODEL_GENERAL
from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import StateFieldError
from backend.processor.import_processor.prompt.model_tagging import (
    MODEL_TAGGING_SYSTEM_PROMPT,
    MODEL_TAGGING_USER_PROMPT_TEMPLATE,
)
from backend.processor.import_processor.state import ImportGraphState

class NodeModelTagging(BaseNode):
    """按 Q3 策略打 product_model / knowledge_type 标签；对 kb_entities 只读（ADR-0002）。"""

    name: str = "node_model_tagging"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        knowledge_type = state.get("knowledge_type")
        if knowledge_type not in KNOWLEDGE_TYPES:
            raise StateFieldError(field_name="knowledge_type", expected_type=str)

        chunks: List[Dict] = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", expected_type=list)

        catalog: List[str] = state.get("model_catalog") or []
        file_model = state.get("product_model", "")

        if file_model:
            canonical = self._normalize(file_model, catalog)
            if canonical == MODEL_GENERAL and catalog:
                self.logger.warning(
                    "文件级机型 %s 不在权威目录中，仍按原值写入（请检查目录）", file_model
                )
                canonical = file_model
            for chunk in chunks:
                chunk["product_model"] = canonical
                chunk["knowledge_type"] = knowledge_type
            self.logger.info("文件级机型标注完成：%s -> %s", file_model, canonical)
        else:
            for idx, chunk in enumerate(chunks, start=1):
                raw = self._tag_chunk(chunk, catalog)
                chunk["product_model"] = self._normalize(raw, catalog)
                chunk["knowledge_type"] = knowledge_type
                if idx % 20 == 0 or idx == len(chunks):
                    self.logger.info("chunk 级标注进度：%s/%s", idx, len(chunks))

        state["chunks"] = chunks
        return state

    def _tag_chunk(self, chunk: Dict, catalog: List[str]) -> str:
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        context = f"标题：{title}\n内容：{content}"[: self.config.model_tagging_max_chars]

        try:
            user_prompt = MODEL_TAGGING_USER_PROMPT_TEMPLATE.format(
                title=title, content=context, catalog=", ".join(catalog) if catalog else "（无）"
            )
            llm = ChatOpenAI(
                model=lm_config.item_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                extra_body={"enable_thinking": False},
            )
            response = llm.invoke([SystemMessage(content=MODEL_TAGGING_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
            raw = response.content.strip().strip('"').strip("'")
            return raw
        except Exception as exc:
            self.logger.error("机型标注 LLM 调用失败：%s，回退 general", exc)
            return MODEL_GENERAL

    @staticmethod
    def _normalize(raw: str, catalog: List[str]) -> str:
        if not raw:
            return MODEL_GENERAL
        text = raw.strip()
        for model in catalog:
            if text.lower() == model.lower():
                return model
        for model in catalog:
            if model.lower() in text.lower() or text.lower() in model.lower():
                return model
        return MODEL_GENERAL
