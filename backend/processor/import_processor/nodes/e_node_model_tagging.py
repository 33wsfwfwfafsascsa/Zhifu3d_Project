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
    """打 product_model / knowledge_type 标签；对 kb_entities 只读。"""

    name: str = "node_model_tagging"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # --- 输入校验 ---
        knowledge_type = state.get("knowledge_type")

        if knowledge_type not in KNOWLEDGE_TYPES:
            raise StateFieldError(field_name="knowledge_type", expected_type=str) # 白名单校验

        chunks: List[Dict] = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", expected_type=list) # d节点必须已切分

        catalog: List[str] = state.get("model_catalog") or [] # 权威目录（可为空）
        file_model = state.get("product_model", "") # 文件级机型（可为空）

        # --- 分支 1：有文件级机型 → 直接写所有 chunk ---
        if file_model:
            canonical = self._normalize(file_model, catalog) # 先归一化
            # 目录非空但匹配不到：警告并按“原值”写入（保留用户意图）
            if canonical == MODEL_GENERAL and catalog:
                self.logger.warning(
                    "文件级机型 %s 不在权威目录中，仍按原值写入（请检查目录）", file_model
                )
                canonical = file_model
            for chunk in chunks:
                chunk["product_model"] = canonical # 全文件同一机型
                chunk["knowledge_type"] = knowledge_type # 知识类型同样直写
            self.logger.info("文件级机型标注完成：%s -> %s", file_model, canonical)
            # --- 分支 2：无文件级机型 → 逐 chunk 调 LLM ---
        else:
            for idx, chunk in enumerate(chunks, start=1):
                raw = self._tag_chunk(chunk, catalog) # LLM 返回原始机型词
                chunk["product_model"] = self._normalize(raw, catalog) # 归一化到目录
                chunk["knowledge_type"] = knowledge_type
                # 每 20 个或最后一个打印进度
                if idx % 20 == 0 or idx == len(chunks):
                    self.logger.info("chunk 级标注进度：%s/%s", idx, len(chunks))

        state["chunks"] = chunks # 写回（chunks 是同一个 list，也可不写，但显式更清晰）
        return state

    def _tag_chunk(self, chunk: Dict, catalog: List[str]) -> str:
        """调用 LLM 判断单个 chunk 的机型，失败回退 general。"""
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        # 上下文截断到 model_tagging_max_chars（默认 2500），防止超长 chunk 撑爆提示词
        context = f"标题：{title}\n内容：{content}"[: self.config.model_tagging_max_chars]

        try:
            # 用模板填充用户提示词；目录为空显示“（无）”
            user_prompt = MODEL_TAGGING_USER_PROMPT_TEMPLATE.format(
                title=title, content=context, catalog=", ".join(catalog) if catalog else "（无）"
            )
            llm = ChatOpenAI(
                model=lm_config.item_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                # 关闭推理模型思考过程，仅拿结论（DashScope 兼容参数）
                extra_body={"enable_thinking": False},
            )
            response = llm.invoke([SystemMessage(content=MODEL_TAGGING_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
            # 清理首尾引号（模型偶尔不遵守“只返回一个词”）
            raw = response.content.strip().strip('"').strip("'")
            return raw
        except Exception as exc:
            # LLM 故障不阻断导入：标注为 general 降级
            self.logger.error("机型标注 LLM 调用失败：%s，回退 general", exc)
            return MODEL_GENERAL

    @staticmethod
    def _normalize(raw: str, catalog: List[str]) -> str:
        """把 LLM/用户输入归一化到权威目录；匹配不到返回 general。"""
        if not raw:
            return MODEL_GENERAL
        text = raw.strip()
        # 第一轮：大小写不敏感的精确匹配（最安全）
        for model in catalog:
            if text.lower() == model.lower():
                return model
        # 第二轮：大小写不敏感的模糊匹配（包含关系）
        for model in catalog:
            if model.lower() in text.lower() or text.lower() in model.lower():
                return model
        return MODEL_GENERAL
