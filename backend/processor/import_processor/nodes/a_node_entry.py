"""入口节点：文件类型校验、标题提取、路由标志。"""

import logging
from pathlib import Path

from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import FileProcessingError, StateFieldError, ValidationError
from backend.processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """任务分发：识别 pdf/md，提取 file_title。"""

    name = "node_entry"

    def process(self, state: ImportGraphState):
        logging.info("node_entry 节点开始执行...")
        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise StateFieldError(field_name="import_file_path", expected_type=str)

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件 {import_file_path_obj.name} 不存在")

        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
            with open(import_file_path_obj, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
        else:
            raise ValidationError(message=f"不支持的文件后缀：{import_file_path_obj.suffix}")

        state["file_title"] = import_file_path_obj.stem

        # 输出目录默认落在 MD_ROOT_DIR/<file_title>，便于人工检查与后续缓存扩展
        if not state.get("file_dir"):
            md_root = self.config.md_root_dir or "."
            state["file_dir"] = str(Path(md_root) / import_file_path_obj.stem)

        return state
