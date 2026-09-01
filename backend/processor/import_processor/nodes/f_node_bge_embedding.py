"""BGE-M3 向量化节点：为 chunk 生成稠密 + 稀疏向量。"""

import logging
from typing import Dict, List

from backend.processor.import_processor.config import MODEL_GENERAL
from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import StateFieldError
from backend.processor.import_processor.state import ImportGraphState
from backend.utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """批量向量化，绑定 dense_vector / sparse_vector。"""

    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", expected_type=list)

        output_data = self._step_2_generate_embeddings(chunks)
        state["chunks"] = output_data
        return state

    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        output_data = []
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i : i + batch_size]
            input_texts = []
            for doc in batch_texts:
                model = doc.get("product_model", "")
                content = doc.get("content", "")
                prefix = f"{model}\n" if model and model != MODEL_GENERAL else ""
                input_texts.append(f"{prefix}{content}")

            docs_embeddings = generate_embeddings(input_texts)
            for j, doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            self.logger.info("已完成 %s-%s 项嵌入", i + 1, min(i + len(batch_texts), len(chunks)))
        return output_data
