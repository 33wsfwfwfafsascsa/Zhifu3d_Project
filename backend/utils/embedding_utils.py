"""BGE-M3 本地混合向量工具：稠密 + 稀疏双向量生成。"""

import logging
import threading

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from backend.config.embedding_config import embedding_config

logger = logging.getLogger(__name__)

# 模型单例，避免重复初始化
_bge_m3_ef = None
_bge_m3_lock = threading.Lock()


def get_bge_m3_ef() -> BGEM3EmbeddingFunction:
    """获取 BGE-M3 单例；模型目录不存在时给出明确报错。"""
    global _bge_m3_ef
    with _bge_m3_lock:
        if _bge_m3_ef is not None:
            return _bge_m3_ef

        model_path = embedding_config.bge_m3_path
        if not model_path:
            raise RuntimeError("BGE_M3_PATH 未配置，请在 .env 中设置本地 BGE-M3 权重目录")

        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_path, # 本地权重目录
            device=embedding_config.bge_device or None,
            use_fp16=embedding_config.bge_fp16,# 半精度省显存
            normalize_embeddings=True,
        )
        return _bge_m3_ef


def generate_embeddings(texts: list[str]) -> dict:
    """批量生成稠密 + 稀疏向量。

    Returns:
        {"dense": [[...], ...], "sparse": [{idx: weight, ...}, ...]}
    """
    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)

    # 把 CSR 格式的 sparse 矩阵切成每段文本的 (indices, data)
    processed_sparse = []
    for i in range(len(texts)):
        # 第 i 段文本在稀疏矩阵中的行区间由 indptr[i] ~ indptr[i+1] 界定
        sparse_indices = embeddings["sparse"].indices[
            embeddings["sparse"].indptr[i] : embeddings["sparse"].indptr[i + 1]
        ].tolist()
        sparse_data = embeddings["sparse"].data[
            embeddings["sparse"].indptr[i] : embeddings["sparse"].indptr[i + 1]
        ].tolist()
        processed_sparse.append(dict(zip(sparse_indices, sparse_data)))

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse,
    }
