"""导入流程自定义异常。"""


class ImportProcessError(Exception):
    """导入流程基础异常。"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)


class StateFieldError(ImportProcessError):
    """状态字段缺失、为空或类型不符。"""

    def __init__(
        self,
        node_name: str = "",
        field_name: str = "",
        expected_type: type = None,
        message: str = "",
        cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f"，期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(ImportProcessError):
    """配置错误。"""


class FileProcessingError(ImportProcessError):
    """文件处理错误。"""


class PdfConversionError(FileProcessingError):
    """PDF 转换错误。"""


class ImageProcessingError(FileProcessingError):
    """图片处理错误。"""


class DocumentSplitError(ImportProcessError):
    """文档切分错误。"""


class EmbeddingError(ImportProcessError):
    """向量化错误。"""


class LLMError(ImportProcessError):
    """LLM 调用错误。"""


class StorageError(ImportProcessError):
    """存储错误。"""


class MilvusError(StorageError):
    """Milvus 存储错误。"""


class MinioError(StorageError):
    """MinIO 存储错误。"""


class ValidationError(ImportProcessError):
    """数据验证错误。"""
