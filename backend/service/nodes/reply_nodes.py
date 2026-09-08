
from backend.service.state import ServiceGraphState
from backend.utils.mongo_history_utils import save_chat_message

# 三条固定话术：注意保持用户可见语气一致
CHITCHAT_REPLY = "您好，我是智服3D 售后助手，可以帮您解答 3D 打印机使用、故障排查、保修政策等问题，请描述您遇到的问题。"
COMPLAINT_REPLY = "非常抱歉给您带来不好的体验，已为您记录，人工客服将尽快跟进处理。"
#ORDER_PLACEHOLDER_REPLY = "订单/物流查询功能将在下一阶段接入，已为您记录相关信息，请稍后咨询人工客服。"


def _reply(state: ServiceGraphState, text: str) -> ServiceGraphState:
    save_chat_message(
        session_id=state.get("session_id", ""),
        role="assistant",
        text=text,
    )
    state["answer"] = text
    return state


def chitchat_reply(state: ServiceGraphState) -> ServiceGraphState:
    """闲聊固定话术：主动引导用户描述业务问题。"""
    return _reply(state, CHITCHAT_REPLY)


def complaint_reply(state: ServiceGraphState) -> ServiceGraphState:
    """投诉占位话术。"""
    return _reply(state, COMPLAINT_REPLY)


#def order_placeholder_reply(state: ServiceGraphState) -> ServiceGraphState:
    """订单/物流占位话术（工具 Agent Day 4 替换）。"""
    return _reply(state, ORDER_PLACEHOLDER_REPLY)
