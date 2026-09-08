"""文档切分节点：标题粗切 + 长切短合。"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter # 递归字符切分器

from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import StateFieldError
from backend.processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """将长 MD 切分为适中 chunk。"""

    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        必要参数：md_content、file_title
        更新参数：chunks
        """
        # [1/7] 读取输入（file_title、md_content）并统一换行符
        content, file_title = self._step_1_get_inputs(state)
        # [2/7] 按 Markdown 标题粗切
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)
        # [3/7] 完全无标题时把全文降级为单章节
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)
        # [4/7] 长章节二次切分 + 短章节合并
        sections = self._step_4_refine_chunks(sections)
        # [5/7] 输出文档切分统计信息
        self._step_5_print_stats(lines_count, sections)
        # [6/7] 备份 chunks.json（在 Milvus 入库前保留中间结果）
        self._step_6_backup(state, sections)
        # [7/7] 写入状态字典
        state["chunks"] = sections
        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        """
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", expected_type=str)
        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", expected_type=str)
        # 兼容 Windows/旧 Mac 换行，避免标题/段落边界判断不一致
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, file_title

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        """
        # 定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = r"^\s*#{1,6}\s+.+"
        lines = content.split("\n")
        sections: List[Dict[str, str]] = [] #章节列表
        title_count = 0 #标题数量
        current_title = "" #当前章节的标题
        current_lines: List[str] = [] #当前标题和下一个标题之间的文本内容
        in_code_block = False #代码块标记：False当前没在代码块中，True当前在代码块中
        code_block_start_marker = None

        def _flush_section():
            """把当前累积的行写入一个 section，并携带标题/父标题上下文。"""
            if not current_lines:
                return
            sections.append(
                {
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title,
                    # 切分时即记录所属章节，供后续仅对同章节碎片做短章合并
                    "parent_title": current_title or "",
                }
            )

        # 逐行遍历，识别标题和普通行以及代码快
        for line in lines:
            striped_line = line.strip()
            # 识别代码块边界 ```、~~~、````、~~~~ 等（至少 3 个连续字符）
            # 使用正则匹配：行首到行尾只有 ` 或 ~ 字符，且数量>=3
            code_block_marker_match = re.match(r"^(`{3,}|~{3,})$", striped_line)
            if code_block_marker_match:
                marker = code_block_marker_match.group(1)
                if not in_code_block:
                    in_code_block = True # 进入代码块，记录开始的标记特征
                    code_block_start_marker = marker
                elif in_code_block and striped_line == code_block_start_marker:
                    in_code_block = False
                    code_block_start_marker = None
                current_lines.append(line)
                continue
            # 只有在代码块之外且匹配标题模式才算标题
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                _flush_section() # 先结算上一节
                current_title = striped_line # 标题去掉首尾空白作为章节名
                current_lines = [current_title] # 标题行同时作为新节第一行
                title_count += 1
            else:
                current_lines.append(line) # 普通行归入当前节

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
        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        """
        if title_count == 0:
            # 无标题情况：替换为单章节，标题为"无标题"
            self.logger.warning("未识别到 MD 标题，全文作为单章节处理：%s", file_title)
            return [{"title": "无标题", "content": content, "file_title": file_title}]
        return sections

    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
        执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
        """
        # 阶段1：切分超长章节 → 所有章节长度控制在最大长度内
        refined_split = []
        for sec in sections:
            # 对每个章节执行超长切分，结果平铺加入列表（避免嵌套）
            refined_split.extend(self._split_long_section(sec))

        # 阶段2：合并过短章节 → 减少碎片化，提升后续检索/大模型调用效果
        final_sections = self._merge_short_sections(refined_split)
        for sec in final_sections:
            # 无父章节的兜底：父章节 = 自身标题
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""
        return final_sections

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        """
        content = section.get("content", "")
        # 长度未超限，无需切分，直接返回原章节（列表格式保持统一）
        if len(content) <= self.config.max_content_length:
            return [section]

        # 提取章节标题，用于组装子Chunk前缀（保留标题上下文）
        title = section.get("title", "")
        # 标题前缀：带空行分隔，与正文区分开
        prefix = f"{title}\n\n" if title else ""
        # 计算正文可用长度：总长度 - 标题前缀长度（避免标题占满Chunk额度）
        available_len = self.config.max_content_length - len(prefix)

        if available_len <= 0:
            self.logger.warning("章节标题过长，无法切分：%s", title[:20])
            return [section] # 标题本身超限：放弃切分，宁长勿错

        # 清理正文重复标题：避免原章节中正文开头重复标题，导致子Chunk内容冗余
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title) :].lstrip()

        # 初始化LangChain递归分割器（核心工具：按优先级分隔符切分，保留语义）
        # separators：分割符优先级（从粗到细），优先按大语义单元切分，最后才硬拆
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            # 分割符优先级：空行(段落)→换行→中文标点→英文标点→空格，最后硬拆（在 chunk_size 位置强制切断）
            # 先用第一个分隔符进行切分，切分后如果某个 Chunk 还是超过 chunk_size，则继续用下一个优先级的分隔符切分
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        # 切分正文并组装子章节（带完整元信息，便于溯源）
        sub_sections = []
        # 遍历切分后的每个文本块，idx 从 1 开始计数
        for idx, chunk in enumerate(splitter.split_text(body), start=1):
            # 清理空内容：跳过切分后的空字符串
            text = chunk.strip()
            if not text:
                continue
            # 组装子Chunk完整内容 = 标题前缀 + 切分后的正文
            full_text = (prefix + text).strip()
            # 子章节元信息：保留父级关联，添加序号，便于后续检索/溯源
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
        """
        【辅助函数】过短章节合并（减少碎片化，提升检索效果）
        核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，避免跨章节合并
        """
        # 边界处理：空列表直接返回，避免后续索引报错
        if not sections:
            return []

        merged_sections = []
        current_chunk = None
        # 初始化：第一个Chunk直接作为当前待合并块
        for sec in sections:
            if current_chunk is None:
                current_chunk = sec
                continue

            # 章标题后无正文（如 FAQ 的一、二、三章标题）→ 并入下一节，避免孤立空 chunk
            if self._is_heading_only(current_chunk):
                current_chunk["content"] = current_chunk["content"].rstrip() + "\n" + sec["content"]
                current_chunk["title"] = sec.get("title") or current_chunk.get("title")
                current_chunk["parent_title"] = sec.get("parent_title", "")
                if "part" in sec:
                    current_chunk["part"] = sec["part"]
                continue # 继续用当前合并结果与再下一节比较

            # 合并条件：1.当前块长度不足阈值 2.与下一块同父标题（同属一个原章节）
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                # 下一节开头可能重复父标题，去掉再拼
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title) :].lstrip()
                    # 合并内容：空行分隔，保证格式整洁
                current_chunk["content"] += "\n\n" + next_content
                # 更新子Chunk序号：保留最新序号，便于溯源
                if "part" in sec:
                    current_chunk["part"] = sec["part"]
            else:
                # 不满足合并条件：将当前块加入结果，切换为新的待合并块
                merged_sections.append(current_chunk)
                current_chunk = sec
        # 循环结束后，将最后一个待合并块加入结果
        if current_chunk is not None:
            merged_sections.append(current_chunk)
        return merged_sections

    @staticmethod
    def _is_heading_only(section: Dict[str, str]) -> bool:
        """判断章节是否只有标题行、无正文。"""
        content = (section.get("content") or "").strip()
        title = (section.get("title") or "").strip()
        if not content or not title:
            return False
        body = content
        if body.startswith(title):
            body = body[len(title) :].strip() # 去掉第一行标题本身
        if not body:
            return True
        # 合并了多级纯标题（如 # 文档标题 + ## 章标题）后，正文仍可能只由标题行构成
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        return bool(lines) and all(re.match(r"^#{1,6}\s+", line) for line in lines)

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        """
        self.logger.info("=" * 30 + " 文档切分统计 " + "=" * 30)# 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("MD 原始总行数：%s，最终 Chunk 数：%s", lines_count, len(sections))

    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        """
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
            # 备份是“尽力而为”，失败只告警不影响主流程
            self.logger.error("Chunk 备份失败：%s", exc, exc_info=False)
