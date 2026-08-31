"""Mock 业务 API 配置：从 .env 读取 MySQL 连接信息。"""

import os

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

load_dotenv()


MYSQL_CONFIG: dict = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3307")),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DB", "zhifu3d"),
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
}


def get_connection() -> pymysql.connections.Connection:
    """创建 MySQL 连接（Mock 阶段每次新建，简单够用）。"""
    return pymysql.connect(**MYSQL_CONFIG)
