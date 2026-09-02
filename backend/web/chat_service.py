"""客服对话 API：POST /api/chat（结构化 JSON 信封，非流式）。

启动：python -m backend.web.chat_service（默认 127.0.0.1:8002）
"""

import logging
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from backend.processor.import_processor.base import setup_logging
from backend.service.main_graph import run
from backend.service.state import ServiceGraphState
from backend.utils.mongo_history_utils import save_chat_message

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


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    """执行客服编排并返回结构化响应信封（Q8 契约）。"""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message 不能为空")

    session_id = request.session_id or str(uuid.uuid4())
    user_message_id = save_chat_message(session_id, "user", message)
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

    return {
        "session_id": session_id,
        "intent": result.get("intent", ""),
        "confidence": result.get("confidence", 0.0),
        "models": result.get("models") or [],
        "needs_model_confirmation": result.get("needs_model_confirmation", False),
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "escalate": result.get("escalate", False),
    }


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app, host="127.0.0.1", port=8002)
