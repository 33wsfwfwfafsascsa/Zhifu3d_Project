"""DashScope TextReRank 重排工具。"""

import dashscope

from backend.config.reranker_config import reranker_config


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    """调用 DashScope TextReRank 重排，返回与 documents 对齐的相关性分数。"""
    if not reranker_config.model:
        raise RuntimeError("TEXT_RERANK_MODEL 未配置，无法调用 Rerank")

    dashscope.api_key = reranker_config.api_key
    call_kwargs: dict = {
        "model": reranker_config.model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
    }
    if reranker_config.instruct and reranker_config.instruct.lower() != "false":
        call_kwargs["instruct"] = reranker_config.instruct
    response = dashscope.TextReRank.call(
        **call_kwargs,
    )

    status_code = response.get("status_code")
    if status_code != 200:
        raise RuntimeError(f"DashScope rerank 调用失败: {response.get('message')}")

    results = response.output.get("results", [])
    scores = [0.0] * len(documents)
    for item in results:
        scores[int(item.get("index"))] = float(item.get("relevance_score"))
    return scores
