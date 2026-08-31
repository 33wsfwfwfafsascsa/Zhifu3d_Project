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
        ("Creality", "Ender-3 S1 Pro", "Ender-3 S1 Pro 3D打印机", "3d_printer", 12),
        ("Creality", "CR-10 SE", "CR-10 SE 3D打印机", "3d_printer", 12),
        ("Creality", "Halot Mage", "Halot Mage 光固化3D打印机", "resin_printer", 12),
        ("Creality", "K1 SE", "K1 SE 3D打印机", "3d_printer", 12),
        ("Creality", "K2-Plus", "K2-Plus 3D打印机", "3d_printer", 12),
        ("Creality", "K2-Pro", "K2-Pro 3D打印机", "3d_printer", 12),
        ("Creality", "CFS", "CFS 多色耗材管理系统", "accessory", 12),
        ("Prusa", "MK4", "Prusa MK4 3D打印机", "3d_printer", 24),
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
        ("ORD-20260801-002", "李雷", "K1C", "K1C 3D打印机", 2399.00, "completed", "2025-07-01 09:00:00"),
        ("ORD-20260820-003", "韩梅梅", "K2", "K2 3D打印机", 3999.00, "pending", "2026-08-20 16:20:00"),
        ("ORD-20260701-004", "陈晨", "K2-Plus", "K2-Plus 3D打印机", 2699.00, "paid", "2026-07-01 11:00:00"),
        ("ORD-20260705-005", "赵磊", "MK4", "MK4 3D打印机", 4999.00, "shipped", "2026-07-05 14:30:00"),
        ("ORD-20260710-006", "孙丽", "K2-Pro", "K2-Pro 3D打印机", 1899.00, "completed", "2026-07-10 09:15:00"),
        ("ORD-20260718-007", "周强", "CFS", "CFS 耗材盒", 1599.00, "refunded", "2026-07-18 16:40:00"),
        ("ORD-20260725-008", "吴敏", "Ender-5 Max", "Ender-5 Max 3D打印机", 2199.00, "paid", "2026-07-25 10:05:00"),
        ("ORD-20260802-009", "郑浩", "Ender-3 S1 Pro", "Ender-3 S1 Pro 3D打印机", 2099.00, "shipped", "2026-08-02 13:20:00"),
        ("ORD-20260808-010", "冯雪", "CR-10 SE", "CR-10 SE 3D打印机", 2499.00, "completed", "2024-08-08 17:00:00"),
        ("ORD-20260812-011", "褚军", "Halot Mage", "Halot Mage 光固化3D打印机", 1499.00, "refunded", "2026-08-12 08:45:00"),
        ("ORD-20260816-012", "卫东", "K1 SE", "K1 SE 3D打印机", 3299.00, "shipped", "2026-08-16 15:10:00"),
        ("ORD-20260822-013", "蒋雯", "Ender-3 V3 Plus", "Ender-3 V3 Plus 3D打印机", 4599.00, "completed", "2026-08-22 19:30:00"),
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
        ("ORD-20260801-002", "2026-08-01 20:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260801-002", "2026-08-02 10:00:00", "深圳转运中心", "运输中"),
        ("ORD-20260801-002", "2026-08-03 15:00:00", "武汉配送站", "派送中"),
        ("ORD-20260801-002", "2026-08-03 18:30:00", "武汉配送站", "已签收"),
        ("ORD-20260820-003", "2026-08-20 17:00:00", "商家仓库", "订单已提交，等待发货"),
        ("ORD-20260701-004", "2026-07-01 12:00:00", "商家仓库", "订单已提交，等待发货"),
        ("ORD-20260705-005", "2026-07-05 15:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260705-005", "2026-07-06 09:30:00", "深圳转运中心", "运输中"),
        ("ORD-20260710-006", "2026-07-10 10:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260710-006", "2026-07-11 14:00:00", "北京转运中心", "运输中"),
        ("ORD-20260710-006", "2026-07-12 11:00:00", "北京配送站", "派送中"),
        ("ORD-20260710-006", "2026-07-12 16:20:00", "北京配送站", "已签收"),
        ("ORD-20260718-007", "2026-07-19 09:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260718-007", "2026-07-20 13:00:00", "广州转运中心", "运输中"),
        ("ORD-20260718-007", "2026-07-21 10:00:00", "广州配送站", "已签收（用户申请退货）"),
        ("ORD-20260725-008", "2026-07-25 11:00:00", "商家仓库", "订单已提交，等待发货"),
        ("ORD-20260802-009", "2026-08-02 14:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260802-009", "2026-08-03 10:30:00", "上海转运中心", "运输中"),
        ("ORD-20260808-010", "2026-08-08 18:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260808-010", "2026-08-09 12:00:00", "成都转运中心", "运输中"),
        ("ORD-20260808-010", "2026-08-10 09:00:00", "成都配送站", "派送中"),
        ("ORD-20260808-010", "2026-08-10 14:30:00", "成都配送站", "已签收"),
        ("ORD-20260812-011", "2026-08-12 09:30:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260812-011", "2026-08-13 15:00:00", "杭州转运中心", "运输中"),
        ("ORD-20260812-011", "2026-08-14 10:30:00", "杭州配送站", "派送中（用户拒收）"),
        ("ORD-20260816-012", "2026-08-16 16:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260816-012", "2026-08-17 11:30:00", "西安转运中心", "运输中"),
        ("ORD-20260822-013", "2026-08-22 20:00:00", "深圳分拣中心", "已揽收"),
        ("ORD-20260822-013", "2026-08-23 10:00:00", "长沙转运中心", "运输中"),
        ("ORD-20260822-013", "2026-08-24 09:30:00", "长沙配送站", "派送中"),
        ("ORD-20260822-013", "2026-08-24 15:00:00", "长沙配送站", "已签收"),
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
            '["喷嘴", "热床", "风扇", "皮带", "PEI打印板"]',
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
        ("TKT-20260818-002", "SESSION-DEMO-002", "咨询", "K1C 切片软件怎么安装", "CZWW Print 一直报安装失败", "K1C", "open", "2026-08-18 16:00:00", None),
        ("TKT-20260819-001", "SESSION-DEMO-003", "故障排查", "K2 打印中断", "打印到一半自动停止，喷嘴温度正常", "K2", "processing", "2026-08-19 10:30:00", "AGENT-001"),
        ("TKT-20260819-002", "SESSION-DEMO-004", "售后", "Ender-3 V3 KE 热床问题退货咨询", "热床不加热，想确认能否退货", "Ender-3 V3 KE", "closed", "2026-08-19 14:00:00", "AGENT-002"),
        ("TKT-20260820-001", "SESSION-DEMO-005", "投诉", "物流太慢要求处理", "下单一周还没收到，强烈不满", "K1 Max", "open", "2026-08-20 09:00:00", None),
        ("TKT-20260820-002", "SESSION-DEMO-006", "咨询", "耗材推荐", "打 PLA 用什么耗材比较好", "Ender-3 V3 KE", "closed", "2026-08-20 15:30:00", "AGENT-001"),
        ("TKT-20260821-001", "SESSION-DEMO-007", "故障排查", "K1C 堵头", "挤出机卡死，加热后还是不出料", "K1C", "processing", "2026-08-21 11:00:00", "AGENT-003"),
        ("TKT-20260822-001", "SESSION-DEMO-008", "售后", "保修查询", "喷嘴坏了是否在保修范围", "Ender-3 V3 KE", "open", "2026-08-22 10:00:00", None),
        ("TKT-20260822-002", "SESSION-DEMO-009", "故障排查", "K2 首层不粘", "底板温度 60 还是粘不住", "K2", "closed", "2026-08-22 17:00:00", "AGENT-002"),
        ("TKT-20260823-001", "SESSION-DEMO-010", "投诉", "客服响应慢投诉", "等了很久没人回复", "K1 Max", "processing", "2026-08-23 09:30:00", "AGENT-003"),
        ("TKT-20260824-001", "SESSION-DEMO-011", "故障排查", "Ender-5 Max 异响", "打印时 X 轴有异响", "Ender-5 Max", "open", "2026-08-24 13:20:00", None),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO tickets (ticket_id, session_id, category, summary, customer_desc, model, status, created_at, agent_id) "
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
