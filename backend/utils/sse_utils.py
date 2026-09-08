"""SSE 流式通道工具——进程内会话队列 + FastAPI 异步生成器。"""

import asyncio
import json
import queue
from typing import Any, AsyncGenerator

from fastapi import Request


class SSEEvent:
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    ESCALATE = "escalate"
    TICKET = "ticket"
    OPERATOR = "operator"
    FINAL = "final"
    ERROR = "error"
    CLOSE = "__close__"

# 会话 ID → 消息队列（进程内；重启/多进程不共享）
_session_stream: dict[str, "queue.Queue"] = {}


def get_sse_queue(session_id: str) -> "queue.Queue | None":
    """取会话队列；不存在返回 None。"""
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建/覆盖会话队列（重复连接会覆盖旧队列，旧连接的事件会丢）。"""
    stream_queue: "queue.Queue" = queue.Queue()
    _session_stream[session_id] = stream_queue
    return stream_queue


def remove_sse_queue(session_id: str) -> None:
    """连接结束移除队列。"""
    _session_stream.pop(session_id, None)


def _sse_pack(event: str, data: dict[str, Any]) -> str:
    """包装成 SSE 协议文本：event: x\ndata: json\n\n。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: dict[str, Any]) -> None:
    """向会话队列写入事件；无队列时静默丢弃（不抛错）。"""
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


def push_progress(session_id: str | None, stage: str, label: str) -> None:
    """推送阶段进度事件；无会话或 SSE 队列时静默忽略。"""
    if session_id:
        push_to_session(session_id, SSEEvent.PROGRESS, {"stage": stage, "label": label})


async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    """FastAPI StreamingResponse 生成器。"""
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        return

    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack(SSEEvent.READY, {})
        while True:
            if await request.is_disconnected():
                break
            try:
                # queue.get 是阻塞调用，放到线程池并带 1s 超时，避免卡死事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue # 超时无消息，继续循环（顺便检查断开）
            if msg.get("event") == SSEEvent.CLOSE:
                break
            yield _sse_pack(msg.get("event"), msg.get("data"))
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        return
    finally:
        remove_sse_queue(session_id)
