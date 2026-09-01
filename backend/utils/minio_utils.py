"""MinIO 客户端工具：连接 + 桶初始化（公共读）。"""

import json
import logging

from minio import Minio

from backend.config.minio_config import minio_config

logger = logging.getLogger(__name__)

minio_client = None


def _init_minio_client() -> Minio | None:
    client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=False,
    )
    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"],
            }
        ],
    }
    client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))
    return client


def get_minio_client() -> Minio | None:
    """获取 MinIO 单例；未配置 endpoint 时返回 None。"""
    global minio_client
    if minio_client is not None:
        return minio_client
    if not minio_config.endpoint:
        logger.warning("MINIO_ENDPOINT 未配置，MinIO 客户端不可用")
        return None
    try:
        minio_client = _init_minio_client()
    except Exception as exc:
        logger.error("MinIO 初始化失败: %s", exc)
        minio_client = None
    return minio_client
