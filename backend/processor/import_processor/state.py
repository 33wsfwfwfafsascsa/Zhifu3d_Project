"""导入流程状态定义。"""

import copy
from typing import List, TypedDict


class ImportGraphState(TypedDict, total=False):
    # 任务标识
    task_id: str
    # 控制标志
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool
    # 路径信息
    import_file_path: str
    file_dir: str
    pdf_path: str
    md_path: str
    # 文件信息
    file_title: str
    # 元数据（Q3/Q4 设计）：文件级知识类型 + 文件级机型
    knowledge_type: str
    product_model: str
    model_catalog: List[str]
    # 中间数据
    md_content: str
    chunks: List


GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "knowledge_type": "",
    "product_model": "",
    "model_catalog": [],
    "md_content": "",
    "chunks": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，支持覆盖。"""
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state
