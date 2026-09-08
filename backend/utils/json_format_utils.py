"""JSON 序列化工具：兼容 MongoDB ObjectId 等特殊类型。"""

import json
from typing import Any

from bson import ObjectId # MongoDB 主键类型


class CustomJSONEncoder(json.JSONEncoder):
    """自定义编码器：ObjectId 转字符串。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    """统一 JSON 序列化入口。"""
    # ensure_ascii=False：中文原样输出（默认），便于日志/调试可读
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, cls=CustomJSONEncoder)
