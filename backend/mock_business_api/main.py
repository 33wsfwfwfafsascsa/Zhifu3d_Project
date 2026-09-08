"""智服3D Mock 业务 API：订单 / 物流 / 退换货政策 / 工单。

接口契约与真实业务系统一致，切换真实系统仅需替换 BASE_URL 与鉴权头。
"""

import json
import uuid
from datetime import date, datetime, timedelta # 保修日期计算
from typing import List, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

try:
    from .config import get_connection
except ImportError:  # 支持脚本方式运行：python backend/mock_business_api/main.py
    from config import get_connection

app = FastAPI(
    title="智服3D-Mock业务API",
    description="订单/物流/退换货政策/工单 Mock 服务（端口 8001）",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketCreate(BaseModel):
    """创建工单请求体。"""
    session_id: str = Field(..., description="关联会话 ID")
    category: str = Field(..., description="咨询/故障/售后/投诉")
    summary: str = Field(..., description="工单摘要")
    customer_desc: str = Field("", description="客户问题描述")
    model: str = Field("", description="关联机型")


class TicketStatusUpdate(BaseModel):
    """状态更新请求体：Literal 限制只能三选一。"""
    status: Literal["open", "processing", "closed"] = Field(..., description="目标状态")


def _db_execute(sql: str, params: tuple = ()) -> List[dict]:
    """统一数据库查询入口：执行 SQL 并返回字典列表。"""
    try:
        conn = get_connection() # 每次新建连接
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()
        return rows
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"数据库操作失败: {exc}") from exc


def _db_commit(sql: str, params: tuple = ()) -> None:
    """统一数据库写入入口：执行 SQL 并提交。"""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"数据库操作失败: {exc}") from exc


def _parse_excluded(raw: Optional[str]) -> List[str]:
    """把 excluded 的 JSON 数组字符串解析为列表，解析失败时返回空列表。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw) # 正常存的是 JSON 数组字符串
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return [x.strip() for x in raw.split("、") if x.strip()]


def _add_months(dt: datetime, months: int) -> datetime:
    """日期加 N 个月（处理月末边界，如 8-31 + 1 月 → 9-30）。"""
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = (date(year, month + 1, 1) - timedelta(days=1)).day
    day = min(dt.day, days_in_month)
    return datetime(year, month, day, dt.hour, dt.minute, dt.second)


@app.get("/health")
def health() -> dict:
    """健康检查。"""
    return {"ok": True}


@app.get("/api/orders/{order_id}")
def get_order(order_id: str) -> dict:
    """订单查询：状态/商品/金额/下单时间。"""
    rows = _db_execute(
        "SELECT order_id, customer_name, product_model, product_name, amount, status, created_at "
        "FROM orders WHERE order_id = %s",
        (order_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="订单不存在")
    return rows[0]


@app.get("/api/logistics/{order_id}")
def get_logistics(order_id: str) -> dict:
    """物流轨迹查询：先确认订单存在，再返回轨迹列表。"""
    order = _db_execute("SELECT order_id FROM orders WHERE order_id = %s", (order_id,))
    if not order:
        raise HTTPException(status_code=404, detail="订单不存在")
    events = _db_execute(
        "SELECT event_time, node, description FROM logistics_events "
        "WHERE order_id = %s ORDER BY event_time",
        (order_id,),
    )
    return {"order_id": order_id, "events": events}


@app.get("/api/refund-policy")
def get_refund_policy(product_type: str) -> dict:
    """退换货政策查询（结构化字段，供精确计算）。"""
    rows = _db_execute(
        "SELECT policy_id, product_type, warranty_months, return_days, exchange_days, "
        "excluded, terms_text, source_url FROM refund_policies WHERE product_type = %s",
        (product_type,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="未找到该产品类型的政策")
    policy = rows[0]
    policy["excluded"] = _parse_excluded(policy.get("excluded"))
    return policy


@app.get("/api/warranty/{order_id}")
def get_warranty(order_id: str, part: Optional[str] = None) -> dict:
    """保修判定：签收日优先，无签收节点退回下单日；支持部件级排除判定。"""
    order = _db_execute(
        "SELECT order_id, product_model, created_at FROM orders WHERE order_id = %s",
        (order_id,),
    )
    if not order:
        raise HTTPException(status_code=404, detail="订单不存在")

    product = _db_execute(
        "SELECT model, warranty_months FROM products WHERE model = %s",
        (order[0]["product_model"],),
    )
    policy = _db_execute(
        "SELECT policy_id, excluded FROM refund_policies WHERE product_type = %s",
        ("3D打印机整机",),
    )

    warranty_months = int(product[0]["warranty_months"]) if product else 12
    policy_id = policy[0]["policy_id"] if policy else None
    excluded = _parse_excluded(policy[0].get("excluded")) if policy else []

    # 签收日优先：取物流中最近一条“已签收”节点时间；没有则退回下单日
    signed_events = _db_execute(
        "SELECT event_time FROM logistics_events "
        "WHERE order_id = %s AND description LIKE %s ORDER BY event_time DESC LIMIT 1",
        (order_id, "%已签收%"),
    )
    if signed_events:
        warranty_start = signed_events[0]["event_time"]
    else:
        warranty_start = order[0]["created_at"]

    # 统一成 datetime（兼容 pymysql 返回 datetime/date/str）
    if isinstance(warranty_start, datetime):
        warranty_start_dt = warranty_start
    elif isinstance(warranty_start, date):
        warranty_start_dt = datetime.combine(warranty_start, datetime.min.time())
    else:
        warranty_start_dt = datetime.strptime(str(warranty_start)[:19], "%Y-%m-%d %H:%M:%S")

    warranty_end_dt = _add_months(warranty_start_dt, warranty_months)
    today = date.today()
    in_warranty = today <= warranty_end_dt.date()
    remaining_days = max((warranty_end_dt.date() - today).days, 0)

    result = {
        "order_id": order_id,
        "product_model": order[0]["product_model"],
        "warranty_start": warranty_start_dt.date().isoformat(),
        "warranty_end": warranty_end_dt.date().isoformat(),
        "warranty_months": warranty_months,
        "in_warranty": in_warranty,
        "remaining_days": remaining_days,
        "policy_id": policy_id,
        "excluded": excluded,
    }
    if part:
        result["part"] = part
        result["part_in_warranty"] = part not in excluded
    return result


@app.post("/api/tickets", status_code=201)
def create_ticket(payload: TicketCreate) -> dict:
    """创建工单：自动生成 ticket_id，初始状态 open。"""
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _db_commit(
        "INSERT INTO tickets (ticket_id, session_id, category, summary, customer_desc, model, status, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, 'open', %s, %s)",
        (
            ticket_id,
            payload.session_id,
            payload.category,
            payload.summary,
            payload.customer_desc,
            payload.model,
            now,
            now,
        ),
    )
    return _db_execute("SELECT * FROM tickets WHERE ticket_id = %s", (ticket_id,))[0]


@app.get("/api/tickets/{ticket_id}")
def get_ticket(ticket_id: str) -> dict:
    """工单查询。"""
    rows = _db_execute("SELECT * FROM tickets WHERE ticket_id = %s", (ticket_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="工单不存在")
    return rows[0]


@app.patch("/api/tickets/{ticket_id}/status")
def update_ticket_status(ticket_id: str, payload: TicketStatusUpdate) -> dict:
    """工单状态流转：open → processing → closed。"""
    rows = _db_execute("SELECT * FROM tickets WHERE ticket_id = %s", (ticket_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="工单不存在")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _db_commit(
        "UPDATE tickets SET status = %s, updated_at = %s WHERE ticket_id = %s",
        (payload.status, now, ticket_id),
    )
    return _db_execute("SELECT * FROM tickets WHERE ticket_id = %s", (ticket_id,))[0]


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
