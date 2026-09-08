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
        # 支持测试时注入自定义配置；未注入则取全局单例
        self.config = config or get_config()
        # 每个节点独立 logger：import.<node_name>，便于按节点过滤日志
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """节点被 LangGraph 调用的入口：包一层开始/完成/失败日志。"""
        try:
            self.logger.info("--- %s 开始 ---", self.name)
            result = self.process(state)
            self.logger.info("--- %s 完成 ---", self.name)
            return result
        except Exception as exc:
            # 统一记录失败并包装：node_name 保留在异常里，便于定位是哪个节点
            self.logger.error("%s 执行失败: %s", self.name, exc)
            raise ImportProcessError(message=str(exc), node_name=self.name, cause=exc) from exc

    @abstractmethod
    def process(self, state: T) -> T:
        """子类必须实现：接收 state，返回更新后的 state。"""
        pass

    def log_step(self, step_name: str, message: str = "") -> None:
        """输出分段日志，格式：[step_name] message。"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO) -> None:
    """给 CLI/脚本配置全局日志格式（时间、logger 名、级别、消息）。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
