"""SSE 流式通道工具：本阶段仅移植备用，Day 4 接入流式输出（PRD 7.2）。"""

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


_session_stream: dict[str, "queue.Queue"] = {}


def get_sse_queue(session_id: str) -> "queue.Queue | None":
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    stream_queue: "queue.Queue" = queue.Queue()
    _session_stream[session_id] = stream_queue
    return stream_queue


def remove_sse_queue(session_id: str) -> None:
    _session_stream.pop(session_id, None)


def _sse_pack(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: dict[str, Any]) -> None:
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


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
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue
            if msg.get("event") == SSEEvent.CLOSE:
                break
            yield _sse_pack(msg.get("event"), msg.get("data"))
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        return
    finally:
        remove_sse_queue(session_id)
