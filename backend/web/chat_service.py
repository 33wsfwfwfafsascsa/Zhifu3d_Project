"""客服对话 API：POST /api/chat（同步信封 / is_stream 后台执行）+ GET /api/stream/{session_id} SSE。
启动：python -m backend.web.chat_service（默认 127.0.0.1:8002）
"""

import logging
import os
import threading # 流式模式后台线程
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.staticfiles import StaticFiles # 前端静态托管
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse # SSE 响应
from starlette.middleware.cors import CORSMiddleware

from backend.processor.import_processor.base import setup_logging
from backend.service.main_graph import run
from backend.service.state import ServiceGraphState
from backend.utils.mongo_history_utils import clear_history, get_recent_messages, save_chat_message
from backend.utils.sse_utils import (
    SSEEvent,# 事件类型常量（ready/progress/delta/.../final）
    create_sse_queue,# 为会话创建队列
    push_to_session,# 往会话队列写事件
    sse_generator,# SSE 异步生成器
)
from backend.utils.session_utils import STATUS_ESCALATED, STATUS_PROCESSING, get_session
from backend.web.agent_routes import router as agent_router

logger = logging.getLogger(__name__)

RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")) # 默认 120 次/分钟
RATE_LIMIT_WINDOW_SECONDS = 60.0
_rate_hits: dict[str, list[float]] = {} # client_ip -> 请求时间戳列表
_rate_lock = threading.Lock() # 多线程保护


app = FastAPI(
    title="智服3D-客服对话API",
    description="意图分类 → 机型确认 → RAG 检索 → 答案生成",
)
app.include_router(agent_router) # 坐席工作台路由
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """/api/chat 超限返回 429；其余路径直接放行。"""
    if request.url.path == "/api/chat" and request.method == "POST":
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _rate_lock:
            if len(_rate_hits) > 1024:  # 防止进程长时间运行后字典无限膨胀
                _rate_hits.clear()
            hits = [t for t in _rate_hits.get(client, []) if now - t < RATE_LIMIT_WINDOW_SECONDS]
            if len(hits) >= RATE_LIMIT_MAX:
                _rate_hits[client] = hits
                return JSONResponse(status_code=429, content={"detail": "操作太频繁，请稍后再试"})
            hits.append(now)
            _rate_hits[client] = hits
    return await call_next(request)


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str | None = Field(None, description="会话ID，首轮可省略由服务端生成")
    is_stream: bool = Field(False, description="是否走 SSE 流式通道")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    """执行客服编排；is_stream=true 时后台执行并返回 session_id，前端经 /api/stream 消费事件。"""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message 不能为空")

    # --- 会话与用户消息落库 ---
    session_id = request.session_id or str(uuid.uuid4())
    # 先落用户消息；Mongo 不可用时返回 ""（后续可能因此丢历史）
    user_message_id = save_chat_message(session_id, "user", message)
    # --- 已转人工/坐席处理中的会话：拦截新对话 ---
    existing = get_session(session_id)
    if existing and existing.get("status") in (STATUS_ESCALATED, STATUS_PROCESSING):
        reply = "您的问题已转接人工客服处理中，请稍候。"
        if request.is_stream:
             # 流式：直接推 final（客户端可能正挂着 SSE）
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {"answer": reply, "status": "escalated", "image_urls": [], "intent": "", "models": [], "citations": []},
            )
        return {
            "session_id": session_id,
            "answer": reply,
            "escalate": True,
            "handled_by_operator": True,
        }
     # --- 流式模式：后台线程执行，接口立即返回 ---
    if request.is_stream:
        threading.Thread(
            target=_run_turn,
            args=(session_id, user_message_id, message),
            daemon=True,
        ).start()
        return {"session_id": session_id, "streaming": True}

    # --- 同步模式：直接跑完整客服主图 ---
    initial_state: ServiceGraphState = {
        "session_id": session_id,
        "user_message_id": user_message_id,
        "original_query": message,
        "enable_web_search": True,
    }
    try:
        result = run(initial_state)
    except Exception as exc:
        logger.exception("客服编排执行失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc

    return _envelope(result)


@app.get("/api/stream/{session_id}")
async def stream(session_id: str, request: Request) -> StreamingResponse:
    """SSE 长连接：前端先连再发消息；坐席 operator/ticket 事件也经此通道推送。"""
    create_sse_queue(session_id)
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history/{session_id}")
def history(session_id: str) -> list[dict]:
    """会话历史（用户/机器人/坐席消息，按时间正序）。"""
    return [
        {"role": msg.get("role"), "text": msg.get("text"), "ts": msg.get("ts")}
        for msg in get_recent_messages(session_id)
    ]


@app.delete("/api/history/{session_id}")
def delete_history(session_id: str) -> dict:
    """清空会话历史。"""
    deleted = clear_history(session_id)
    return {"session_id": session_id, "deleted": deleted}


def _run_turn(session_id: str, user_message_id: str, message: str) -> None:
    """后台执行编排并推送 delta/final；escalate 事件由节点内推送。"""
    try:
        result = run(
            {
                "session_id": session_id,
                "user_message_id": user_message_id,
                "original_query": message,
                "is_stream": True,
                "enable_web_search": True,
            }
        )
    except Exception as exc:
        logger.exception("后台编排执行失败: %s", exc)
        push_to_session(session_id, SSEEvent.ERROR, {"error": f"处理失败: {exc}"})
        push_to_session(session_id, SSEEvent.FINAL, {"answer": "", "status": "error", "image_urls": []})
        return

    citations = result.get("citations") or []
    image_urls = list(dict.fromkeys(img for c in citations for img in (c.get("image_urls") or [])))
    push_to_session(
        session_id,
        SSEEvent.FINAL,
        {
            "answer": result.get("answer", ""),
            "status": "escalated" if result.get("escalate") else "ok",
            "image_urls": image_urls,
            "intent": result.get("intent", ""),
            "models": result.get("models") or [],
            "citations": citations,
        },
    )


def _envelope(result: ServiceGraphState) -> dict:
    """同步响应信封。"""
    return {
        "session_id": result.get("session_id", ""),
        "intent": result.get("intent", ""),
        "confidence": result.get("confidence", 0.0),
        "models": result.get("models") or [],
        "needs_model_confirmation": result.get("needs_model_confirmation", False),
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "escalate": result.get("escalate", False),
        "escalate_reason": result.get("escalate_reason", ""),
    }

# 静态前端：当前文件在 backend/web，父级两级 = 项目根
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="127.0.0.1", port=8002) # 对话 + 坐席 + 前端
