"""知识库导入 API：文件上传（带元数据）+ 任务进度查询。

启动：python -m backend.web.import_service（默认 127.0.0.1:8000）
"""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from starlette.middleware.cors import CORSMiddleware

from backend.config.minio_config import minio_config
from backend.processor.import_processor.base import setup_logging
from backend.processor.import_processor.catalog import load_model_catalog
from backend.processor.import_processor.config import KNOWLEDGE_TYPES, get_config
from backend.processor.import_processor.main_graph import KBImportWorkflow
from backend.processor.import_processor.state import create_default_state
from backend.utils.minio_utils import get_minio_client
from backend.utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="智服3D-知识库导入API", description="文件上传触发导入流水线 + 任务进度查询")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_graph_task(task_id: str, file_dir: str, import_file_path: str,
                   knowledge_type: str, product_model: str) -> None:
    """后台执行 LangGraph 导入流水线，实时更新任务状态。"""
    try:
        update_task_status(task_id, "processing")
        state = create_default_state(
            task_id=task_id,
            file_dir=file_dir,
            import_file_path=import_file_path,
            knowledge_type=knowledge_type,
            product_model=product_model,
            model_catalog=load_model_catalog(),
        )
        workflow = KBImportWorkflow()
        for event in workflow.graph.stream(state, stream_mode="updates"):
            for node_name in event:
                add_done_task(task_id, node_name)
        update_task_status(task_id, "completed")
        set_task_result(task_id, "file", import_file_path)
    except Exception as exc:
        logger.exception("导入任务失败：%s", task_id)
        update_task_status(task_id, "failed")
        set_task_result(task_id, "error", str(exc))


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/catalog")
def get_catalog() -> dict:
    """标准机型目录（ModelCatalog），供上传页「适用机型」下拉选择。"""
    return {"code": 200, "models": load_model_catalog()}


@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    knowledge_type: str = Form(...),
    product_model: str = Form(""),
):
    """上传单个 PDF/MD 并触发导入；返回 task_id 供轮询进度。"""
    if knowledge_type not in KNOWLEDGE_TYPES:
        raise HTTPException(status_code=422, detail=f"knowledge_type 必须为 {sorted(KNOWLEDGE_TYPES)}")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".md"):
        raise HTTPException(status_code=422, detail="仅支持 .pdf / .md 文件")

    safe_name = Path(file.filename or "upload").name
    task_id = str(uuid.uuid4())
    config = get_config()
    data_root = config.data_root_dir or "data/raw"
    md_root = config.md_root_dir or "data/processed"

    # 原始文件留在 raw/<日期>/<task_id>；流水线产物统一落 processed/<文件标题>
    upload_dir = Path(data_root) / datetime.now().strftime("%Y%m%d") / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    import_file_path = upload_dir / safe_name
    work_dir = Path(md_root) / Path(safe_name).stem

    add_running_task(task_id, "upload_file")
    try:
        with open(import_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        update_task_status(task_id, "failed")
        set_task_result(task_id, "error", str(exc))
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc

    # PDF：解析节点会把产物写进 work_dir；MD：先复制到 work_dir 再跑，保证产物不落 raw
    pipeline_path = import_file_path
    if suffix == ".md":
        work_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = work_dir / safe_name
        shutil.copy2(import_file_path, pipeline_path)

    # 原始文件备份到 MinIO（失败不阻断本地导入）
    try:
        minio_client = get_minio_client()
        if minio_client:
            object_name = f"pdf_files/{datetime.now().strftime('%Y%m%d')}/{safe_name}"
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=str(import_file_path),
                content_type=file.content_type or "application/octet-stream",
            )
    except Exception as exc:
        logger.warning("原文件上传 MinIO 失败（继续本地导入）：%s", exc)

    add_done_task(task_id, "upload_file")
    background_tasks.add_task(
        run_graph_task, task_id, str(work_dir), str(pipeline_path), knowledge_type, product_model
    )
    return {"code": 200, "task_id": task_id, "knowledge_type": knowledge_type, "product_model": product_model}


@app.get("/api/status/{task_id}")
def get_task_progress(task_id: str) -> dict:
    """查询导入任务进度与状态。"""
    status = get_task_status(task_id)
    return {
        "code": 200,
        "task_id": task_id,
        "status": status,
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
        "error": get_task_result(task_id, "error") if status == "failed" else "",
    }


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="127.0.0.1", port=8000)
