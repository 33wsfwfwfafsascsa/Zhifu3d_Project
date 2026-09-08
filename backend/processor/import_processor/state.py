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
    import_file_path: str # 用户上传/命令行传入的原始文件路径
    file_dir: str
    pdf_path: str
    md_path: str
    # 文件信息
    file_title: str
    # 元数据：文件级知识类型 + 文件级机型
    knowledge_type: str # manual / faq / troubleshooting / policy
    product_model: str # 文件级机型；为空表示按 chunk 级 LLM 标注
    model_catalog: List[str] # MySQL products 去重后的权威机型目录（供归一化）
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
    """创建默认状态，支持覆盖。
    1. deepcopy 默认值：避免调用方修改影响模块级 GRAPH_DEFAULT_STATE；
    2. update(overrides)：把调用方传入的 task_id / import_file_path 等字段覆盖进去。
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state
