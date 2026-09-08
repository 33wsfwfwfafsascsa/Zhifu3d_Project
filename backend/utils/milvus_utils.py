"""Milvus 客户端工具：连接、集合建表、混合检索。"""

import json
import logging

from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker

from backend.config.milvus_config import milvus_config

logger = logging.getLogger(__name__)

_milvus_client = None


def get_milvus_client() -> MilvusClient | None:
    """获取 Milvus 单例；连接失败返回 None。"""
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    if not milvus_config.milvus_url:
        logger.warning("MILVUS_URL 未配置，Milvus 客户端不可用")
        return None
    try:
        _milvus_client = MilvusClient(uri=milvus_config.milvus_url)
    except Exception as exc:
        logger.error("Milvus 连接失败: %s", exc)
        _milvus_client = None
    return _milvus_client


def escape_milvus_string(value: str) -> str:
    """转义过滤表达式中的字符串特殊字符（防注入/防语法破坏）。"""
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


def build_chunk_filter_expr(
    models: list[str] | None = None,
    knowledge_types: list[str] | None = None,
) -> str | None:
    """构造 kb_chunks 过滤表达式（product_model / knowledge_type in [...]）。"""
    parts: list[str] = []
    if models:
        # 用 json.dumps 生成带引号的安全字符串（比手工拼引号稳）
        quoted = ", ".join(json.dumps(m, ensure_ascii=False) for m in models)
        parts.append(f"product_model in [{quoted}]")
    if knowledge_types:
        quoted = ", ".join(json.dumps(t, ensure_ascii=False) for t in knowledge_types)
        parts.append(f"knowledge_type in [{quoted}]")
    # 两段条件用 and 连接；全空返回 None 表示不过滤
    return " and ".join(parts) if parts else None


def create_kb_chunks_collection(client: MilvusClient, collection_name: str, vector_dim: int) -> None:
    """创建 kb_chunks 集合：标量字段含 product_model 与 knowledge_type。"""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="product_model", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="knowledge_type", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

    index_params = client.prepare_index_params()
    # 稠密索引：AUTOINDEX + COSINE（BGE 已归一化，COSINE 合适）
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    # 稀疏索引：倒排 + IP
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"},
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def create_kb_entities_collection(client: MilvusClient, collection_name: str, vector_dim: int) -> None:
    """创建 kb_entities 集合：标准机型目录（model + 双向量）。"""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="model", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"},
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5,
):
    """构建稠密 + 稀疏混合检索请求。"""
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,
    )
    return [dense_req, sparse_req]


def hybrid_search(
    client: MilvusClient,
    collection_name: str,
    reqs,
    ranker_weights=(0.5, 0.5),
    limit=5,
    output_fields=None,
    search_params=None,
):
    """执行稠密 + 稀疏混合检索；失败返回 None。"""
    try:
        rerank = WeightedRanker(*ranker_weights)
        if output_fields is None:
            output_fields = ["product_model"]
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )
        logger.info("Milvus 混合搜索完成，集合[%s]共 %s 条", collection_name, len(res[0]))
        return res
    except Exception as exc:
        logger.error("Milvus 混合搜索失败，集合[%s]：%s", collection_name, exc, exc_info=True)
        return None
