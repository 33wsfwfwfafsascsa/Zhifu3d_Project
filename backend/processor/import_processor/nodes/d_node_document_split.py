"""文档切分节点：标题粗切 + 长切短合。"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import StateFieldError
from backend.processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """将长 MD 切分为适中 chunk。"""

    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        content, file_title = self._step_1_get_inputs(state)
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)
        sections = self._step_4_refine_chunks(sections)
        self._step_5_print_stats(lines_count, sections)
        self._step_6_backup(state, sections)
        state["chunks"] = sections
        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", expected_type=str)
        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", expected_type=str)
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, file_title

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        title_pattern = r"^\s*#{1,6}\s+.+"
        lines = content.split("\n")
        sections: List[Dict[str, str]] = []
        title_count = 0
        current_title = ""
        current_lines: List[str] = []
        in_code_block = False
        code_block_start_marker = None

        def _flush_section():
            if not current_lines:
                return
            sections.append(
                {"title": current_title, "content": "\n".join(current_lines), "file_title": file_title}
            )

        for line in lines:
            striped_line = line.strip()
            code_block_marker_match = re.match(r"^(`{3,}|~{3,})$", striped_line)
            if code_block_marker_match:
                marker = code_block_marker_match.group(1)
                if not in_code_block:
                    in_code_block = True
                    code_block_start_marker = marker
                elif in_code_block and striped_line == code_block_start_marker:
                    in_code_block = False
                    code_block_start_marker = None
                current_lines.append(line)
                continue

            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(line)

        _flush_section()
        self.logger.info(
            "按标题粗切完成，共 %s 个章节，标题 %s 个，总行数 %s",
            len(sections),
            title_count,
            len(lines),
        )
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(
        self, content: str, sections: List[Dict[str, str]], title_count: int, file_title: str
    ) -> List[Dict[str, str]]:
        if title_count == 0:
            self.logger.warning("未识别到 MD 标题，全文作为单章节处理：%s", file_title)
            return [{"title": "无标题", "content": content, "file_title": file_title}]
        return sections

    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        refined_split = []
        for sec in sections:
            refined_split.extend(self._split_long_section(sec))

        final_sections = self._merge_short_sections(refined_split)
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""
        return final_sections

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        content = section.get("content", "")
        if len(content) <= self.config.max_content_length:
            return [section]

        title = section.get("title", "")
        prefix = f"{title}\n\n" if title else ""
        available_len = self.config.max_content_length - len(prefix)
        if available_len <= 0:
            self.logger.warning("章节标题过长，无法切分：%s", title[:20])
            return [section]

        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title) :].lstrip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        sub_sections = []
        for idx, chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue
            full_text = (prefix + text).strip()
            sub_sections.append(
                {
                    "title": f"{title}-{idx}" if title else f"chunk-{idx}",
                    "content": full_text,
                    "parent_title": title,
                    "part": idx,
                    "file_title": section.get("file_title"),
                }
            )
        return sub_sections

    def _merge_short_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not sections:
            return []

        merged_sections = []
        current_chunk = None
        for sec in sections:
            if current_chunk is None:
                current_chunk = sec
                continue

            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")
            if is_current_short and is_same_parent:
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title) :].lstrip()
                current_chunk["content"] += "\n\n" + next_content
                if "part" in sec:
                    current_chunk["part"] = sec["part"]
            else:
                merged_sections.append(current_chunk)
                current_chunk = sec

        if current_chunk is not None:
            merged_sections.append(current_chunk)
        return merged_sections

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        self.logger.info("=" * 30 + " 文档切分统计 " + "=" * 30)
        self.logger.info("MD 原始总行数：%s，最终 Chunk 数：%s", lines_count, len(sections))

    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        try:
            md_path = state.get("md_path")
            if not md_path:
                return
            backup_path = Path(md_path).parent / "chunks.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info("Chunk 备份成功：%s", backup_path)
        except Exception as exc:
            self.logger.error("Chunk 备份失败：%s", exc, exc_info=False)
