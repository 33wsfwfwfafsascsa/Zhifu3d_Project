"""BGE-M3 向量化节点：为 chunk 生成稠密 + 稀疏向量。"""

import logging
from typing import Dict, List

from backend.processor.import_processor.config import MODEL_GENERAL # general 不加机型前缀
from backend.processor.import_processor.base import BaseNode
from backend.processor.import_processor.exceptions import StateFieldError
from backend.processor.import_processor.state import ImportGraphState
from backend.utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """批量向量化，绑定 dense_vector / sparse_vector。"""

    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 前置校验：e 节点必须已产出 chunks
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", expected_type=list)

        output_data = self._step_2_generate_embeddings(chunks) # 分批向量化
        state["chunks"] = output_data # 用新列表替换
        return state

    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        批量生成向量（核心业务逻辑）
        核心逻辑：
            1. 分批处理：避免一次性处理过多数据导致显存溢出（OOM）。
            2. 向量生成：调用模型批量生成 Dense（稠密）和 Sparse（稀疏）向量。
        """
        # 初始化空列表，存储最终带向量的文本切片
        output_data = []
        # 默认 8，控制显存/批吞吐
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i : i + batch_size]
            input_texts = []
            for doc in batch_texts:
                model = doc.get("product_model", "") # e 节点标注的机型
                content = doc.get("content", "")
                # 机型作为语义前缀，帮助模型区分“不同机型下的同一表述”；
                # general 不加前缀，避免污染通用知识
                prefix = f"{model}\n" if model and model != MODEL_GENERAL else ""
                input_texts.append(f"{prefix}{content}")

            # 一次调用同时返回 dense/sparse（BGE-M3 混合向量）
            docs_embeddings = generate_embeddings(input_texts)
            for j, doc in enumerate(batch_texts):
                item = doc.copy() # 浅拷贝：保留原字段，再挂向量
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            self.logger.info("已完成 %s-%s 项嵌入", i + 1, min(i + len(batch_texts), len(chunks)))
        return output_data
