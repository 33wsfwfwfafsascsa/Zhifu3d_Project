"""客服编排主图：单 Agent 最小闭环（非流式，Day 3 口径）。"""

from langgraph.graph import END, StateGraph

from backend.processor.query_processor.nodes.a_node_model_confirm import NodeModelConfirm
from backend.service.nodes.intent_classifier import IntentClassifier
from backend.service.nodes.rag_agent import RagAgent
from backend.service.nodes.reply_nodes import (
    chitchat_reply,
    complaint_reply,
)
from backend.service.nodes.tool_agent import ToolAgent
from backend.service.state import ServiceGraphState

intent_classifier = IntentClassifier()
model_confirm_node = NodeModelConfirm()
rag_agent = RagAgent()
tool_agent = ToolAgent()


def _route_after_intent(state: ServiceGraphState) -> str:
    """意图路由：闲聊/投诉直接回复；订单实体或售后意图走工具 Agent；其余走机型确认。"""
    intent = state.get("intent", "")
    if intent == "chitchat":
        return "chitchat_reply"
    if intent == "complaint":
        return "complaint_reply"
    if state.get("order_ids") or intent == "after_sales":
        return "tool_agent"
    return "model_confirm"


def _route_after_confirm(state: ServiceGraphState) -> str:
    """机型未确认（反问/兜底）直接结束，否则进入检索。"""
    return "rag_agent" if not state.get("needs_model_confirmation") else "__end__"


def build_service_graph():
    """构建并编译客服编排状态图。"""
    workflow = StateGraph(ServiceGraphState)
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("chitchat_reply", chitchat_reply)
    workflow.add_node("complaint_reply", complaint_reply)
    workflow.add_node("tool_agent", tool_agent)
    workflow.add_node("model_confirm", model_confirm_node)
    workflow.add_node("rag_agent", rag_agent)

    workflow.set_entry_point("intent_classifier")
    workflow.add_conditional_edges(
        "intent_classifier",
        _route_after_intent,
        {
            "chitchat_reply": "chitchat_reply",
            "complaint_reply": "complaint_reply",
            "tool_agent": "tool_agent",
            "model_confirm": "model_confirm",
        },
    )
    for node_name in ("chitchat_reply", "complaint_reply", "tool_agent"):
        workflow.add_edge(node_name, END)
    workflow.add_conditional_edges(
        "model_confirm",
        _route_after_confirm,
        {"rag_agent": "rag_agent", "__end__": END},
    )
    workflow.add_edge("rag_agent", END)
    return workflow.compile()


service_graph = build_service_graph()


def run(state: ServiceGraphState) -> ServiceGraphState:
    """执行客服编排，返回最终状态。"""
    return service_graph.invoke(state)
