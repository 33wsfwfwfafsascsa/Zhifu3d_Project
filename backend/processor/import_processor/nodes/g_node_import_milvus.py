"""Milvus 入库节点：建表 + 幂等清理 + 批量插入。"""

import logging
from typing import Any, Dict, List

from backend.config.milvus_config import milvus_config
from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import MilvusError, StateFieldError
from backend.processor.import_processor.state import ImportGraphState
from backend.utils.milvus_utils import create_kb_chunks_collection, escape_milvus_string, get_milvus_client


class NodeImportMilvus(BaseNode):
    """切片数据入库；幂等键 = file_title + knowledge_type。"""

    name: str = "node_import_milvus"

    @staticmethod
    def _truncate_utf8(value: str, max_bytes: int) -> str:
        """按 UTF-8 字节数截断（Milvus VARCHAR 长度按字节计，中文需按字节截）。"""
        if not value:
            return ""
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks, vector_dimension = self._step_1_check_input(state)
        client = self._step_2_prepare_collection(vector_dimension)
        self._step_3_clean_old_data(client, chunks)
        updated_chunks = self._step_4_insert_data(client, chunks)
        state["chunks"] = updated_chunks
        return state

    def _step_1_check_input(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", expected_type=list)

        first_chunk = chunks[0]
        if "dense_vector" not in first_chunk:
            raise StateFieldError(field_name="chunks", message="缺失 dense_vector 字段", expected_type=list)
        if "sparse_vector" not in first_chunk:
            raise StateFieldError(field_name="chunks", message="缺失 sparse_vector 字段", expected_type=list)
        if "product_model" not in first_chunk or "knowledge_type" not in first_chunk:
            raise StateFieldError(field_name="chunks", message="缺失 product_model/knowledge_type 标签", expected_type=list)

        vector_dimension = len(first_chunk["dense_vector"])
        return chunks, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension: int):
        client = get_milvus_client()
        if not client:
            raise MilvusError("Milvus 连接失败")

        collection_name = milvus_config.chunks_collection
        if not client.has_collection(collection_name):
            self.logger.info("集合不存在，创建 %s", collection_name)
            create_kb_chunks_collection(client, collection_name, vector_dimension)
            client.load_collection(collection_name)
        return client

    def _step_3_clean_old_data(self, client, chunks: List[Dict[str, Any]]) -> None:
        file_title = chunks[0].get("file_title")
        knowledge_type = chunks[0].get("knowledge_type")
        if not file_title or not knowledge_type:
            raise StateFieldError(field_name="chunks", message="file_title/knowledge_type 不能为空", expected_type=list)

        safe_title = escape_milvus_string(file_title)
        safe_type = escape_milvus_string(knowledge_type)
        try:
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title=='{safe_title}' and knowledge_type=='{safe_type}'",
            )
            self.logger.info("已清理旧数据：file_title=%s, knowledge_type=%s", file_title, knowledge_type)
        except Exception as exc:
            raise MilvusError(f"Milvus 数据删除失败: {exc}") from exc

    def _step_4_insert_data(self, client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()
            if "part" not in item_copy:
                item_copy["part"] = 0
            # 对齐 schema 的 VARCHAR 长度限制（按 UTF-8 字节计）
            item_copy["title"] = self._truncate_utf8(item_copy.get("title") or "", 100)
            item_copy["parent_title"] = self._truncate_utf8(item_copy.get("parent_title") or "", 100)
            item_copy["file_title"] = self._truncate_utf8(item_copy.get("file_title") or "", 100)
            item_copy["product_model"] = self._truncate_utf8(item_copy.get("product_model") or "", 100)
            item_copy["knowledge_type"] = self._truncate_utf8(item_copy.get("knowledge_type") or "", 50)
            item_copy["content"] = self._truncate_utf8(item_copy.get("content") or "", 65535)
            data_to_insert.append(item_copy)

        insert_result = client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
        insert_count = insert_result.get("insert_count", 0)
        inserted_ids = insert_result.get("ids", [])
        if inserted_ids:
            for idx, item in enumerate(chunks_json_data):
                item["chunk_id"] = str(inserted_ids[idx])
        self.logger.info("批量插入完成：%s 条", insert_count)
        return chunks_json_data
