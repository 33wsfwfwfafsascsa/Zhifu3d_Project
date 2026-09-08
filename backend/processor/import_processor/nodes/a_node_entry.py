"""入口节点：文件类型校验、标题提取、路由标志。"""

import logging
from pathlib import Path # 路径对象：用于后缀判断（.suffix）、主文件名提取（.stem）、存在性检查

from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import FileProcessingError, StateFieldError, ValidationError
from backend.processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """任务分发：识别 pdf/md，提取 file_title。"""
    # 类属性 name 主要影响日志名（import.node_entry）；
    # 注意：与主图注册的节点 key "a_node_entry" 并不相同

    name = "node_entry"

    def process(self, state: ImportGraphState):
        # [1/6] 开始日志：使用全局 logging.info，而不是 self.logger（一致性问题，见文末注意事项）
        logging.info("node_entry 节点开始执行...")

        # [2/6] 从共享状态读取待导入文件路径；为空则抛 StateFieldError 中断流程
        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise StateFieldError(field_name="import_file_path", expected_type=str)
        
        # [3/6] 转为 Path 对象并校验文件真实存在；不存在则抛 FileProcessingError
        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件 {import_file_path_obj.name} 不存在")

        # [4/6] 按后缀分流（严格小写匹配）：
        if import_file_path_obj.suffix == ".pdf":
            # PDF 分支：只标记“需要 PDF 读取”，不在此处读内容；
            # 真实转换由下一个节点 b_node_pdf_to_md 完成
            state["is_pdf_read_enabled"] = True # 路由标志：主图据此走 b 节点
            state["pdf_path"] = import_file_path # 记录 PDF 原始路径供下游读取
        elif import_file_path_obj.suffix == ".md":
            # Markdown 分支：标记“需要 MD 读取”，并同步把全文读入内存
            state["is_md_read_enabled"] = True # 路由标志：主图据此直接走 c 节点
            state["md_path"] = import_file_path # 记录 MD 原始路径
            # 一次性读全文（utf-8），供后续图片处理/文档切分使用；
            # 大文件时这里会占较多内存，可优化

            with open(import_file_path_obj, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
        else:
            # 其他后缀（含大写 .PDF/.Md）：直接校验失败，终止本次导入
            raise ValidationError(message=f"不支持的文件后缀：{import_file_path_obj.suffix}")

        # [5/6] 用 Path.stem 提取主文件名（不含扩展名），作为知识库中的文件标题/标识
        state["file_title"] = import_file_path_obj.stem

        # [6/6] 输出目录兜底：调用方未显式指定 file_dir 时，
        # 默认放到 <md_root_dir>/<file_title>，方便人工检查中间产物
        if not state.get("file_dir"):
            md_root = self.config.md_root_dir or "."
            state["file_dir"] = str(Path(md_root) / import_file_path_obj.stem)
        # LangGraph 节点约定：返回 state（或其子集）作为更新结果
        return state
