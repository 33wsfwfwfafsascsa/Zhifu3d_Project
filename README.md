# 智服3D · 企业级智能客服系统

基于 RAG + LangGraph 多 Agent 的 3D 打印机售后智能客服系统（毕设/求职作品集项目）。

## 需求与技术方案

本仓库根目录：《企业级智能客服系统-PRD及技术方案.md》

> 说明：原始文档位于 `D:\ZG_ZhiKu_Project\docs\`，本仓库保存一份副本用于自包含交付；修改时以原始文档为准，避免双源漂移。

## 目录结构

```text
zhifu3d/
├── backend/
│   ├── service/               # 客服编排（LangGraph 三 Agent）
│   ├── mock_business_api/     # Mock 订单/物流/政策/工单（8001）
│   ├── middleware/            # 限流/降级
│   └── web/                   # FastAPI 路由 + 静态前端
├── frontend/                  # 单前端双视图（user/agent）
├── data/
│   ├── raw/                   # manuals/ faq/ policy/ troubleshooting/
│   └── processed/             # 清洗后的切片/问答对
├── eval/                      # 评测集 + 评测脚本
├── 企业级智能客服系统-PRD及技术方案.md
├── docker-compose.yml         # （待建）
├── .env.example               # 环境变量模板
└── README.md
```

## 当前阶段

目录骨架已建立，代码与语料采集未开始。
