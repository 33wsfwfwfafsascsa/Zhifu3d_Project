"""MongoDB 配置：会话历史存储。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MongoConfig:
    url: str
    db_name: str


mongo_config = MongoConfig(
    url=os.getenv("MONGO_URL", ""),
    db_name=os.getenv("MONGO_DB_NAME", "zhifu3d"),
)
