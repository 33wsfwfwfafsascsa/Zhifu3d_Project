"""查询流程节点基类：统一入口、日志与异常处理。"""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

logger = logging.getLogger(__name__) # 查询模块共享一个 logger，不按节点拆分

StateT = TypeVar("StateT") # 状态类型参数；实际常被绑定为 QueryGraphState


class NodeBase(ABC, Generic[StateT]):
    """节点统一接口，子类覆盖 name 并实现 process。"""

    name: str = "base_node"

    def __call__(self, state: StateT) -> StateT:
        """可调用入口：LangGraph / rag_agent 直接以 node(state) 方式执行。"""
        logger.info("--- %s 开始 ---", self.name)
        try:
            result = self.process(state)
            logger.info("--- %s 完成 ---", self.name)
            return result
        except Exception as exc:
            # 记录完整堆栈后原样上抛，不包装（由调用方决定降级策略）
            logger.error("%s 执行失败: %s", self.name, exc, exc_info=True)
            raise

    @abstractmethod
    def process(self, state: StateT) -> StateT:
        """节点核心逻辑，子类实现。"""
