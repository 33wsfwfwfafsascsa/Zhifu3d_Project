"""客服对话 API：POST /api/chat（同步信封 / is_stream 后台执行）+ GET /api/stream/{session_id} SSE。

启动：python -m backend.web.chat_service（默认 127.0.0.1:8002）
"""

import logging
import threading
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from backend.processor.import_processor.base import setup_logging
from backend.service.main_graph import run
from backend.service.state import ServiceGraphState
from backend.utils.mongo_history_utils import save_chat_message
from backend.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    push_to_session,
    sse_generator,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="智服3D-客服对话API",
    description="意图分类 → 机型确认 → RAG 检索 → 答案生成（非流式最小闭环）",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
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

    session_id = request.session_id or str(uuid.uuid4())
    user_message_id = save_chat_message(session_id, "user", message)
    if request.is_stream:
        threading.Thread(
            target=_run_turn,
            args=(session_id, user_message_id, message),
            daemon=True,
        ).start()
        return {"session_id": session_id, "streaming": True}

    initial_state: ServiceGraphState = {
        "session_id": session_id,
        "user_message_id": user_message_id,
        "original_query": message,
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


def _run_turn(session_id: str, user_message_id: str, message: str) -> None:
    """后台执行编排并推送 delta/final；escalate 事件由节点内推送。"""
    try:
        result = run(
            {
                "session_id": session_id,
                "user_message_id": user_message_id,
                "original_query": message,
                "is_stream": True,
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
    """同步响应信封（Q8 契约）。"""
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


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="127.0.0.1", port=8002)
