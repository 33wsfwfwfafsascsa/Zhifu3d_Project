"""客服编排常量：意图标签、knowledge_type 映射与规则兜底。"""

import re

# 五类意图的“合法值”白名单（tuple 不可变，防止被意外修改）
# 注意：这里只定义标签，不定义优先级；路由优先级由 main_graph._route_after_intent 决定
INTENT_LABELS = ("consult", "troubleshoot", "after_sales", "complaint", "chitchat")

# 用户可见的统一等待文案：内部 stage 码仅用于日志/调试，不外显给终端用户
# 无论底层是 understanding/searching/querying/generating，前端都只展示这一句
USER_WAIT_LABEL = "正在思考中，请稍候…"

KNOWLEDGE_TYPE_BY_INTENT: dict[str, list[str]] = {
    "troubleshoot": ["troubleshooting", "faq", "manual"],# 故障优先查排障手册，再 FAQ/手册
    "consult": ["manual", "faq"], # 咨询优先查手册，再 FAQ
    "after_sales": ["policy"],# 售后只查政策
}

# 订单号格式：ORD- 开头，后接至少 4 位字母/数字/连字符
# \b 防止把 ABCORD-123 误判；IGNORECASE 兼容 ord-xxx；findall 返回原始大小写文本
ORDER_ID_PATTERN = re.compile(r"\bORD-[A-Za-z0-9-]{4,}\b", re.IGNORECASE)

# 投诉关键词：最高优先级；注意“我要退款”被归为投诉，不是售后政策查询
COMPLAINT_KEYWORDS = ("投诉","举报","差评","非常不满意","太差了","垃圾","我要退款","割韭菜","骗子","体验极差","太烂了","欺骗消费者","叫你们领导来")
# 售后/保修关键词：触发 after_sales；“退货”是短词，可能误伤“退货流程怎么操作”这类咨询
AFTER_SALES_KEYWORDS = ("保修", "质保", "退换货", "退货", "换货", "退款政策")
# 寒暄关键词：需与“整句长度 ≤12”配合，防止长业务句被误判成闲聊
CHITCHAT_KEYWORDS = ("你好", "您好", "在吗", "谢谢", "再见", "你是谁","你是AI吗")


def rule_based_classify(text: str) -> tuple[str, float, list[str]]:
    """LLM 不可用时的规则兜底分类。"""
    # [分支 1] 投诉关键词命中 → complaint（0.95）
    # 放在最前面：投诉类情绪必须最优先识别，不能落入 consult
    if any(keyword in text for keyword in COMPLAINT_KEYWORDS):
        return "complaint", 0.95, []

    # [分支 2] 正则抽取订单号
    # dict.fromkeys(...) 的作用是“去重且保持首次出现顺序”
    order_ids = list(dict.fromkeys(ORDER_ID_PATTERN.findall(text)))
    if order_ids:
        return "consult", 0.9, order_ids # 有订单号一律归 consult：由 main_graph 据此路由到 tool_agent

    # [分支 3] 售后关键词命中 → after_sales（0.9）
    # 放在 chitchat 之前：例如“在吗，我想问下保修”必须识别为售后而不是闲聊
    if any(keyword in text for keyword in AFTER_SALES_KEYWORDS):
        return "after_sales", 0.9, []

    # [分支 4] 寒暄：要求整句较短（≤12 字符）且命中关键词
    # 防止“你好，我的打印机堵头了怎么解决”这种带业务的长句被误判为闲聊
    stripped = text.strip()
    if len(stripped) <= 12 and any(keyword in stripped for keyword in CHITCHAT_KEYWORDS):
        return "chitchat", 0.9, []

    # [分支 5] 默认兜底：低置信 consult（0.5）
    # 注意：规则兜底无法产出 troubleshoot——LLM 挂了时故障类问题只能走 consult
    return "consult", 0.5, []
