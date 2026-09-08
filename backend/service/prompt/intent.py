"""意图分类提示词。"""

# 系统角色：只允许 JSON 输出，减少模型解释性废话
INTENT_SYSTEM_PROMPT = "你是 3D 打印机售后客服的意图分类器，只输出 JSON。"

# 用户模板：classifier 节点填充 {query}
INTENT_USER_TEMPLATE = """
请对以下用户问题分类并提取订单号。

可选意图：
- consult：产品使用与咨询（如：切片软件设置、耗材推荐、打印参数设置、固件升级、机器使用教程、订单物流等）
- troubleshoot：打印故障与硬件异常（如：堵头、翘边、拉丝、调平失败、层错位、不粘热床、断料、打印机报错代码等）
- after_sales：售后政策与维保（如：保修期查询、退换货申请、配件寄修等）
- complaint：投诉与强强烈情绪泄愤（如：质量太差要退一赔三、投诉到消协、骂人等）
- chitchat：问候与无意义闲聊（如：你好、谢谢、你是谁等）

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
