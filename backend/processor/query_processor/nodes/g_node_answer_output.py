"""答案生成节点：组装 Prompt → LLM 生成 → 写入历史（非流式）。"""

import logging
import re

from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT
from backend.processor.query_processor.state import QueryGraphState
from backend.utils.llm_utils import get_llm_client
from backend.utils.mongo_history_utils import save_chat_message
from backend.utils.sse_utils import SSEEvent, push_to_session

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000


class NodeAnswerOutput(NodeBase[QueryGraphState]):
    """答案生成：基于 reranked_docs + 历史 + 机型组装提示词。"""

    name: str = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        prompt = self._construct_prompt(state)
        answer = self._generate(
            prompt,
            session_id=state.get("session_id"),
            is_stream=bool(state.get("is_stream")),
        )
        state["answer"] = answer
        save_chat_message(
            session_id=state.get("session_id", "default"),
            role="assistant",
            text=answer,
            models=state.get("models") or [],
            image_urls=self._extract_images_from_docs(state.get("reranked_docs") or []),
        )
        return state

    def _construct_prompt(self, state: QueryGraphState) -> str:
        question = state.get("rewritten_query") or state.get("original_query", "")
        context_str, _ = self._format_reranked_docs(state.get("reranked_docs") or [])
        history_str, _ = self._format_chat_history(state.get("history") or [])
        models_str = ", ".join(state.get("models") or []) or "未指定机型"
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str or "暂无历史对话",
            models=models_str,
            question=question,
        )

    def _generate(self, prompt: str, session_id: str | None = None, is_stream: bool = False) -> str:
        try:
            client = get_llm_client()
            if is_stream and session_id:
                return self._generate_stream(client, prompt, session_id)
            response = client.invoke(prompt)
            return str(response.content or "").strip()
        except Exception as exc:
            logger.error("答案生成失败: %s", exc)
            return "抱歉，生成回答时出现错误，请稍后重试或转人工客服。"

    def _generate_stream(self, client, prompt: str, session_id: str) -> str:
        """流式生成并逐块推送 delta；异常时降级单次调用推单条 delta。"""
        chunks: list[str] = []
        try:
            for chunk in client.stream(prompt):
                text = self._chunk_text(chunk)
                if text:
                    chunks.append(text)
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": text})
        except Exception as exc:
            logger.warning("流式生成失败，降级单次调用: %s", exc)
            if not chunks:
                response = client.invoke(prompt)
                text = str(response.content or "").strip()
                if text:
                    chunks.append(text)
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": text})
        return "".join(chunks).strip()

    @staticmethod
    def _chunk_text(chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(str(block.get("text") or ""))
            return "".join(parts)
        return ""

    def _format_reranked_docs(self, reranked_docs: list[dict]) -> tuple[str, int]:
        lines: list[str] = []
        used_chars = 0
        for idx, doc in enumerate(reranked_docs, start=1):
            meta_tags = [f"[{idx}]"]
            for field, template in (
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ):
                field_value = str(doc.get(field) or "").strip()
                if field_value:
                    meta_tags.append(template.format(field_value))
            entry = " ".join(meta_tags) + "\n" + str(doc.get("content") or "")
            if used_chars + len(entry) > MAX_CONTEXT_CHARS:
                break
            lines.append(entry)
            used_chars += len(entry) + 2
        return "\n\n".join(lines), MAX_CONTEXT_CHARS - used_chars

    def _format_chat_history(self, chat_history: list[dict]) -> tuple[str, int]:
        lines: list[str] = []
        used_chars = 0
        role_label_map = {"user": "用户", "assistant": "助手"}
        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue
            line = f"{role_label_map[role]}: {text}"
            if used_chars + len(line) > MAX_CONTEXT_CHARS:
                break
            lines.append(line)
            used_chars += len(line) + 1
        return "\n".join(lines), MAX_CONTEXT_CHARS - used_chars

    def _extract_images_from_docs(self, docs: list[dict]) -> list[str]:
        """从文档 URL 字段与 Markdown 图片语法中提取图片 URL（去重）。"""
        images: list[str] = []
        seen: set[str] = set()
        md_img_pattern = re.compile(r"!\[.*?\]\((.*?\.(?:png|jpg|jpeg|gif|webp|bmp|svg))\)")
        for doc in docs:
            url = str(doc.get("url") or "").strip()
            if url and url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")):
                if url not in seen:
                    seen.add(url)
                    images.append(url)
            for img_url in md_img_pattern.findall(str(doc.get("content") or "")):
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    seen.add(img_url)
                    images.append(img_url)
        return images
