"""导入流程配置：集中管理，支持环境变量覆盖。"""

import os
from dataclasses import dataclass, field
from typing import Optional, Set

from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_TYPES = frozenset({"manual", "faq", "troubleshooting", "policy"})
MODEL_GENERAL = "general"


@dataclass
class ImportConfig:
    # 文档切分
    max_content_length: int = 2000 # 单 chunk 目标最大字符数
    img_content_length: int = 200 # 图片相关内容的长度上限（预留，d 节点未直接用）
    min_content_length: int = 500 # 短 chunk 合并阈值（小于该长度且同父章节时合并）
    overlap_sentences: int = 1
    # 多机型文件 chunk 级标注上下文上限
    model_tagging_max_chars: int = 2500

    # 允许作为文档图片处理的扩展名（c 节点使用）
    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # LLM
    # ===== LLM 相关（注意：节点代码实际更常读 lm_config/mineru_config 等全局配置）=====
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    vl_model: str = field(default_factory=lambda: os.getenv("VL_MODEL", ""))
    item_model: str = field(default_factory=lambda: os.getenv("ITEM_MODEL", ""))
    default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", ""))

    # Milvus
    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "kb_chunks"))
    entity_collection: str = field(default_factory=lambda: os.getenv("ENTITY_COLLECTION", "kb_entities"))

    # MinIO
    minio_endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", ""))
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", ""))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", ""))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", ""))
    minio_secure: bool = False

    # 向量
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    embedding_batch_size: int = 8

    # 数据目录
    data_root_dir: str = field(default_factory=lambda: os.getenv("DATA_BASED_ROOT_DIR", ""))
    md_root_dir: str = field(default_factory=lambda: os.getenv("MD_ROOT_DIR", ""))

    # 图片总结 API 速率限制
    requests_per_minute: int = 15

    @classmethod
    def from_env(cls) -> "ImportConfig":
        return cls()


_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例：首次调用创建，之后复用。"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
