-- 智服3D Mock 业务库表结构（与 PRD 8.5 数据模型一致）

CREATE TABLE IF NOT EXISTS products (
  product_id INT AUTO_INCREMENT PRIMARY KEY,
  brand VARCHAR(50) NOT NULL,
  model VARCHAR(100) NOT NULL UNIQUE,
  name VARCHAR(200) NOT NULL,
  category VARCHAR(50) NOT NULL DEFAULT '3d_printer',
  warranty_months INT NOT NULL DEFAULT 12,
  specs JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS orders (
  order_id VARCHAR(50) PRIMARY KEY,
  customer_name VARCHAR(100) NOT NULL,
  product_model VARCHAR(100) NOT NULL,
  product_name VARCHAR(200) NOT NULL,
  amount DECIMAL(10,2) NOT NULL,
  status VARCHAR(20) NOT NULL COMMENT 'pending/paid/shipped/completed/refunded',
  created_at DATETIME NOT NULL,
  INDEX idx_orders_model (product_model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS logistics_events (
  id INT AUTO_INCREMENT PRIMARY KEY,
  order_id VARCHAR(50) NOT NULL,
  event_time DATETIME NOT NULL,
  node VARCHAR(200) NOT NULL,
  description VARCHAR(500),
  INDEX idx_logistics_order (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS tickets (
  ticket_id VARCHAR(50) PRIMARY KEY,
  session_id VARCHAR(100),
  category VARCHAR(50) COMMENT '咨询/故障/售后/投诉',
  summary VARCHAR(500),
  customer_desc TEXT,
  model VARCHAR(100),
  status VARCHAR(20) NOT NULL DEFAULT 'open' COMMENT 'open/processing/closed',
  agent_id VARCHAR(50),
  created_at DATETIME NOT NULL,
  updated_at DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS refund_policies (
  policy_id INT AUTO_INCREMENT PRIMARY KEY,
  product_type VARCHAR(100) NOT NULL,
  warranty_months INT NOT NULL,
  return_days INT NOT NULL,
  exchange_days INT NOT NULL,
  excluded VARCHAR(500),
  terms_text TEXT,
  source_url VARCHAR(500)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
