"""导入流程节点基类：统一的日志与异常包装。"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from backend.processor.import_processor.config import ImportConfig, get_config
from backend.processor.import_processor.exceptions import ImportProcessError

T = TypeVar("T")


class BaseNode(ABC):
    """所有导入节点继承此基类，实现 process 方法。"""

    name: str = "base_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        try:
            self.logger.info("--- %s 开始 ---", self.name)
            result = self.process(state)
            self.logger.info("--- %s 完成 ---", self.name)
            return result
        except Exception as exc:
            self.logger.error("%s 执行失败: %s", self.name, exc)
            raise ImportProcessError(message=str(exc), node_name=self.name, cause=exc) from exc

    @abstractmethod
    def process(self, state: T) -> T:
        pass

    def log_step(self, step_name: str, message: str = "") -> None:
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
