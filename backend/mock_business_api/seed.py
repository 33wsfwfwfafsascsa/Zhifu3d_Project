"""Mock 业务数据种子脚本：建表 + 清空旧数据 + 插入演示数据。"""

from pathlib import Path
from typing import Dict

try:
    from .config import get_connection
except ImportError:  # 支持脚本方式运行：python backend/mock_business_api/seed.py
    from config import get_connection

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def execute_schema(conn) -> None:
    """逐条执行 schema.sql（pymysql 默认不支持多语句，按分号拆分）。"""
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    conn.commit()


def clear_tables(conn) -> None:
    """清空演示数据（先删子表，避免外键/依赖顺序问题）。"""
    tables = ["logistics_events", "orders", "tickets", "products", "refund_policies"]
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"DELETE FROM {table}")
    conn.commit()


def insert_products(conn) -> int:
    products = [
        ("Creality", "Ender-3 V3 KE", "Ender-3 V3 KE 3D打印机", "3d_printer", 12),
        ("Creality", "Ender-3 V3 Plus", "Ender-3 V3 Plus 3D打印机", "3d_printer", 12),
        ("Creality", "Ender-5 Max", "Ender-5 Max 3D打印机", "3d_printer", 12),
        ("Creality", "K1 Max", "K1 Max 3D打印机", "3d_printer", 12),
        ("Creality", "K1C", "K1C 3D打印机", "3d_printer", 12),
        ("Creality", "K2", "K2 3D打印机", "3d_printer", 12),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO products (brand, model, name, category, warranty_months) "
            "VALUES (%s, %s, %s, %s, %s)",
            products,
        )
    conn.commit()
    return len(products)


def insert_orders(conn) -> int:
    orders = [
        ("ORD-20260815-001", "王小明", "Ender-3 V3 KE", "Ender-3 V3 KE 3D打印机", 1799.00, "shipped", "2026-08-15 10:30:00"),
        ("ORD-20260801-002", "李雷", "K1C", "K1C 3D打印机", 2399.00, "completed", "2026-08-01 09:00:00"),
        ("ORD-20260820-003", "韩梅梅", "K2", "K2 3D打印机", 3999.00, "pending", "2026-08-20 16:20:00"),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO orders (order_id, customer_name, product_model, product_name, amount, status, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            orders,
        )
    conn.commit()
    return len(orders)


def insert_logistics(conn) -> int:
    events = [
        ("ORD-20260815-001", "2026-08-15 18:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260815-001", "2026-08-16 09:00:00", "深圳转运中心", "运输中"),
        ("ORD-20260815-001", "2026-08-17 14:00:00", "武汉转运中心", "到达中转站"),
        ("ORD-20260815-001", "2026-08-18 11:00:00", "武汉配送站", "派送中"),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO logistics_events (order_id, event_time, node, description) "
            "VALUES (%s, %s, %s, %s)",
            events,
        )
    conn.commit()
    return len(events)


def insert_policies(conn) -> int:
    policies = [
        (
            "3D打印机整机",
            12,
            7,
            15,
            "易损件、赠品、消耗品",
            "中国大陆（含港澳台）整机质保 12 个月；电商平台购买支持 7 天无理由退货；"
            "因品质问题非人为损坏，签收后 15 天内可申请退换货；易损件、赠品、消耗品除外。",
            "http://www.creality.cn/index.php/policy.html",
        ),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO refund_policies (product_type, warranty_months, return_days, exchange_days, excluded, terms_text, source_url) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            policies,
        )
    conn.commit()
    return len(policies)


def insert_tickets(conn) -> int:
    tickets = [
        (
            "TKT-20260818-001",
            "SESSION-DEMO-001",
            "故障排查",
            "Ender-3 V3 KE 首层翘边",
            "打印首层总是翘边，试过调平还是不行",
            "Ender-3 V3 KE",
            "open",
            "2026-08-18 15:00:00",
            None,
        ),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO tickets (ticket_id, session_id, category, summary, customer_desc, model, status, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            tickets,
        )
    conn.commit()
    return len(tickets)


def main() -> None:
    conn = get_connection()
    try:
        execute_schema(conn)
        clear_tables(conn)
        counts: Dict[str, int] = {
            "products": insert_products(conn),
            "orders": insert_orders(conn),
            "logistics": insert_logistics(conn),
            "policies": insert_policies(conn),
            "tickets": insert_tickets(conn),
        }
        print(
            f"seed done: products {counts['products']} / orders {counts['orders']} / "
            f"logistics {counts['logistics']} / policies {counts['policies']} / tickets {counts['tickets']}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
