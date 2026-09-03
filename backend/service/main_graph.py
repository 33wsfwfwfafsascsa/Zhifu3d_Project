"""客服编排主图：单 Agent 最小闭环（非流式，Day 3 口径）。"""

from langgraph.graph import END, StateGraph

from backend.processor.query_processor.nodes.a_node_model_confirm import NodeModelConfirm
from backend.service.nodes.escalation import EXPLICIT_KEYWORDS, escalation_check, escalation_output
from backend.service.nodes.intent_classifier import IntentClassifier
from backend.service.nodes.rag_agent import RagAgent
from backend.service.nodes.reply_nodes import chitchat_reply
from backend.service.nodes.tool_agent import ToolAgent
from backend.service.state import ServiceGraphState

intent_classifier = IntentClassifier()
model_confirm_node = NodeModelConfirm()
rag_agent = RagAgent()
tool_agent = ToolAgent()


def _route_after_intent(state: ServiceGraphState) -> str:
    """意图路由：闲聊直接回复；投诉走转人工检查；订单实体/售后走工具 Agent；其余走机型确认。"""
    intent = state.get("intent", "")
    if intent == "chitchat":
        return "chitchat_reply"
    if intent == "complaint":
        return "escalation_check"
    if any(keyword in state.get("original_query", "") for keyword in EXPLICIT_KEYWORDS):
        return "escalation_check"
    if state.get("order_ids") or intent == "after_sales":
        return "tool_agent"
    return "model_confirm"


def _route_after_confirm(state: ServiceGraphState) -> str:
    """机型未确认（反问/兜底）直接结束，否则进入检索。"""
    return "rag_agent" if not state.get("needs_model_confirmation") else "__end__"


def _route_after_check(state: ServiceGraphState) -> str:
    """命中转人工触发条件则进入输出节点，否则结束。"""
    return "escalation_output" if state.get("escalate_reason") else "__end__"


def build_service_graph():
    """构建并编译客服编排状态图。"""
    workflow = StateGraph(ServiceGraphState)
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("chitchat_reply", chitchat_reply)
    workflow.add_node("tool_agent", tool_agent)
    workflow.add_node("model_confirm", model_confirm_node)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("escalation_check", escalation_check)
    workflow.add_node("escalation_output", escalation_output)

    workflow.set_entry_point("intent_classifier")
    workflow.add_conditional_edges(
        "intent_classifier",
        _route_after_intent,
        {
            "chitchat_reply": "chitchat_reply",
            "escalation_check": "escalation_check",
            "tool_agent": "tool_agent",
            "model_confirm": "model_confirm",
        },
    )
    workflow.add_edge("chitchat_reply", END)
    workflow.add_edge("tool_agent", "escalation_check")
    workflow.add_conditional_edges(
        "model_confirm",
        _route_after_confirm,
        {"rag_agent": "rag_agent", "__end__": END},
    )
    workflow.add_edge("rag_agent", "escalation_check")
    workflow.add_conditional_edges(
        "escalation_check",
        _route_after_check,
        {"escalation_output": "escalation_output", "__end__": END},
    )
    workflow.add_edge("escalation_output", END)
    return workflow.compile()


service_graph = build_service_graph()


def run(state: ServiceGraphState) -> ServiceGraphState:
    """执行客服编排，返回最终状态。"""
    return service_graph.invoke(state)
