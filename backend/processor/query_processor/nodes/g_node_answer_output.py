"""答案生成节点：组装 Prompt → LLM 生成 → 写入历史（非流式）。"""

import logging
import re

from backend.processor.query_processor.base import NodeBase
from backend.processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT, NOT_FOUND_REPLY
from backend.processor.query_processor.state import QueryGraphState
from backend.utils.llm_utils import get_llm_client
from backend.utils.mongo_history_utils import save_chat_message
from backend.utils.sse_utils import SSEEvent, push_to_session

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000
MD_IMG_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]*?\.(?:png|jpg|jpeg|gif|webp|bmp|svg))\)", re.IGNORECASE)
BARE_IMG_PATTERN = re.compile(r"https?://\S+?\.(?:png|jpg|jpeg|gif|webp|bmp|svg)", re.IGNORECASE)
REFS_TAG_PATTERN = re.compile(r"<refs>[\d,\s]*</refs>")


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
        answer, refs = self._extract_refs(answer)
        answer = self._sanitize_images(answer, state.get("reranked_docs") or [], refs)
        state["answer"] = answer
        save_chat_message(
            session_id=state.get("session_id", "default"),
            role="assistant",
            text=answer,
            models=state.get("models") or [],
            image_urls=self._extract_images_from_docs(state.get("reranked_docs") or []),
        )
        return state

    @staticmethod
    def _extract_refs(answer: str) -> tuple[str, list[int] | None]:
        """提取并移除文末 <refs> 声明；声明缺失/非法时返回 None（退化为存在性过滤）。"""
        refs: list[int] | None = None
        out_lines: list[str] = []
        for line in answer.splitlines():
            match = re.fullmatch(r"\s*<refs>\s*([\d,\s]*)\s*</refs>\s*", line)
            if match:
                if refs is None:
                    parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
                    parsed = [int(part) for part in parts if part.isdigit()]
                    if parts and len(parsed) == len(parts):
                        refs = parsed
                continue
            out_lines.append(line)
        return "\n".join(out_lines).strip(), refs

    def _sanitize_images(self, answer: str, reranked_docs: list[dict], refs: list[int] | None) -> str:
        """只保留「已声明采用的条目」里的图片，数量不限。

        不删改参考原文：仅对答案里选出的图片做「归属」收口，剔除不属于所采用条目的图片。
        """
        if not answer:
            return answer
        if refs is not None:
            entries = [reranked_docs[i - 1] for i in refs if 1 <= i <= len(reranked_docs)]
        else:
            entries = reranked_docs
        allowed = set(self._extract_images_from_docs(entries))

        answer = MD_IMG_PATTERN.sub(
            lambda match: match.group(0) if match.group(1).strip() in allowed else "",
            answer,
        )
        out_lines: list[str] = []
        for raw in answer.splitlines():
            line = raw.strip()
            if not line:
                out_lines.append(raw)
                continue
            if line == "【图片】":
                continue
            urls = [u.strip() for u in BARE_IMG_PATTERN.findall(line)]
            if urls and not any(u in allowed for u in urls) and re.fullmatch(r"https?://\S+", line):
                continue
            out_lines.append(raw)
        return "\n".join(out_lines)

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
            not_found_reply=NOT_FOUND_REPLY,
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
        """流式生成并逐块推送 delta；<refs> 行在推送前剔除，用户不可见。"""
        chunks: list[str] = []
        buf = ""
        try:
            for chunk in client.stream(prompt):
                text = self._chunk_text(chunk)
                if not text:
                    continue
                chunks.append(text)
                buf += text
                clean, buf = self._redact_refs_lines(buf)
                if clean:
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": clean})
        except Exception as exc:
            logger.warning("流式生成失败，降级单次调用: %s", exc)
            if not chunks:
                response = client.invoke(prompt)
                text = str(response.content or "").strip()
                if text:
                    chunks.append(text)
                    buf += text
        if buf:
            clean, _ = self._redact_refs_lines(buf + "\n")
            if clean:
                push_to_session(session_id, SSEEvent.DELTA, {"delta": clean})
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

    @staticmethod
    def _redact_refs_lines(buf: str) -> tuple[str, str]:
        """逐行冲刷缓冲：剔除行内完整的 <refs> 标记，未换行的尾部留在缓冲继续拼装。"""
        out_lines: list[str] = []
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            out_lines.append(REFS_TAG_PATTERN.sub("", line))
        if not out_lines:
            return "", buf
        return "\n".join(out_lines) + "\n", buf

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
