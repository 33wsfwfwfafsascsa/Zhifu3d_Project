"""内存态任务追踪（单进程）：导入/查询任务进度。"""

from typing import Dict, List

# 四个全局字典：task_id → 对应状态数据
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict[str, str]] = {}

# 任务状态常量
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点英文名 → 前端展示中文名
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
    """惰性初始化任务条目（幂等）。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    """英文节点名转中文；未知节点原样返回。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    """把节点加入 running（去重）。"""
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    """节点完成：从 running 移除并加入 done（去重）。"""
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """写任务结果（如 file / error）。"""
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """读任务结果；注意：未知 task_id 也会被 _ensure_task 创建空条目。"""
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """读总状态；未知任务返回空字符串（不会创建条目）。"""
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """读已完成节点（中文名，保持顺序）。"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_running_task_list(task_id: str) -> List[str]:
    """读运行中节点（中文名）。"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def update_task_status(task_id: str, status_name: str) -> None:
    """更新总状态。"""
    _tasks_status[task_id] = status_name


def clear_task(task_id: str) -> None:
    """删除任务全部记录（当前 import_service 未调用）。"""
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
