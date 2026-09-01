"""内存态任务追踪（单进程）：导入/查询任务进度。"""

from typing import Dict, List

_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "a_node_entry": "检查文件",
    "b_node_pdf_to_md": "PDF转Markdown",
    "c_node_md_img": "Markdown图片处理",
    "d_node_document_split": "文档切分",
    "e_node_model_tagging": "机型标注",
    "f_node_bge_embedding": "向量生成",
    "g_node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
}


def _ensure_task(task_id: str) -> None:
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)


def set_task_result(task_id: str, key: str, value: str) -> None:
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_running_task_list(task_id: str) -> List[str]:
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def update_task_status(task_id: str, status_name: str) -> None:
    _tasks_status[task_id] = status_name


def clear_task(task_id: str) -> None:
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
