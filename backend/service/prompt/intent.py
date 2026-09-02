"""意图分类提示词。"""

INTENT_SYSTEM_PROMPT = "你是 3D 打印机售后客服的意图分类器，只输出 JSON。"

INTENT_USER_TEMPLATE = """
请对以下用户问题分类并提取订单号。

可选意图：
- consult：咨询（使用、参数、切片耗材、订单物流等）
- troubleshoot：故障排查（翘边、堵头、调平等打印异常）
- after_sales：售后政策（保修、退换货）
- complaint：投诉、强烈不满
- chitchat：闲聊

用户问题：
{query}

直接返回 JSON：
{{
  "intent": "troubleshoot",
  "confidence": 0.92,
  "order_ids": ["ORD-20260815-001"]
}}
要求：order_ids 按正则提取，无则空列表；confidence 为 0~1。
"""
