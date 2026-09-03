# 智服3D · 企业级智能客服系统

基于 RAG + LangGraph 多 Agent 的 3D 打印机售后智能客服系统（毕设/求职作品集项目）。

## 需求与技术方案

本仓库根目录：《企业级智能客服系统-PRD及技术方案.md》

> 说明：原始文档位于 `D:\ZG_ZhiKu_Project\docs\`，本仓库保存一份副本用于自包含交付；修改时以原始文档为准，避免双源漂移。

## 目录结构

```text
zhifu3d/
├── backend/
│   ├── service/               # 客服编排（意图/工具/RAG/转人工 + smoke）
│   ├── processor/             # 导入与查询流水线（LangGraph）
│   ├── mock_business_api/     # Mock 订单/物流/政策/工单（8001）
│   ├── config/ utils/         # 配置与工具层
│   └── web/                   # FastAPI 路由（chat/import/agent）+ 静态前端
├── frontend/                  # 单前端双视图（user/agent）
├── data/
│   ├── raw/                   # manuals/ faq/ policy/ troubleshooting/
│   └── processed/             # 清洗后的切片/问答对
├── eval/                      # 评测集 + 评测脚本
├── tests/                     # pytest（Mock API）
├── docs/adr/                  # 决策记录
├── 企业级智能客服系统-PRD及技术方案.md
├── docker-compose.yml         # Mongo/MySQL/MinIO/Milvus
├── .env.example               # 环境变量模板
└── README.md
```

## 快速启动

```powershell
# 1. 依赖容器（Mongo/MySQL/MinIO/Milvus）
docker compose up -d

# 2. 服务进程（可按需启动）
python -m backend.mock_business_api.main   # 8001 Mock 业务 API
python -m backend.web.chat_service         # 8002 对话 + SSE + 坐席 API + 前端

# 3. 冒烟（依赖 8001 在跑）
python -m backend.service.smoke_chat       # 期望 SMOKE RESULT: 9/9 PASS
```

打开 http://127.0.0.1:8002/ 即单页双视图演示控制台（左侧用户聊天、右侧坐席工作台）。

## 当前阶段

- Day 3 已完成：知识导入、检索闭环（POST /api/chat）、80 条评测集与首轮指标。
- Day 4 已完成：工具 Agent（订单/物流/政策/保修）、五类转人工判定、SSE 流式、坐席工作台、前端双视图。
- Day 5 计划：全量 80 条评测 + 自动解决率调优（检索命中率 ≥90%、回答准确率 ≥85%、自动解决率 ≥70%）。

## 评测

```powershell
python -m eval.run_eval                       # 全量（依赖 8001），产出 report.json / badcases.json
python -m eval.run_eval --only eval_066,eval_073   # 定向跑，产出 report.partial.json（不覆盖正式报告）
```
