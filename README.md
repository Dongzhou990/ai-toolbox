# AI 工具箱 SaaS

> 全栈 AI 应用平台 | Agent 自动化研究 · RAG 文档问答 · 多 Agent 协作 · AI 口碑管理

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 在线体验

打开浏览器访问 `http://localhost:8080`

## ✨ 四大功能

| 功能 | 说明 | 技术亮点 |
|------|------|----------|
| 🔍 **Agent 研究** | 输入话题，AI 自动搜索 → 分析 → 生成报告 | ReAct 模式，LLM 自主决策 + 工具调用 |
| 🤝 **多 Agent 协作** | 研究员搜集资料 → 写手生成专业报告 | 双 Agent 分工，协同完成复杂任务 |
| 📄 **RAG 问答** | 上传文档 → 智能分块 → 精准问答 | TF-IDF 向量化 + 余弦相似度检索 |
| 💬 **口碑助手** | 粘贴评价 → 选风格 → AI 秒回 | 8 种风格，差评/好评全覆盖 |

## 🛠 技术架构

```
前端: 原生 HTML/CSS/JS (Apple 风格)
后端: Python FastAPI
AI:   DeepSeek API (可替换 GPT/Claude)
数据库: SQLite (用户/知识库/历史)
检索: TF-IDF + 余弦相似度
Agent: 自研 ReAct 循环 + 多 Agent 协作
```

## 🏃 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
export OPENAI_API_KEY=你的key

# 3. 启动
python server.py

# 4. 打开浏览器
open http://localhost:8080
```

## 📁 项目结构

```
web-app/
├── server.py          # FastAPI 后端（所有 API）
├── database.py        # SQLite 数据层
├── multi_agent.py     # 多 Agent 协作系统
├── review_ai.py       # 口碑助手 · AI 回复生成
├── static/index.html  # 前端界面
├── research-agent/    # Agent 模块
├── rag-qa/           # RAG 模块
└── requirements.txt
```

## 🔑 核心实现

### ReAct Agent 循环

```
用户给目标 → LLM 决策(搜索/读网页/回答) → 执行工具 → 结果回传 → 再决策 → 直到完成
```

### RAG 检索增强生成

```
文档 → 分块 → TF-IDF 向量化 → 问题向量化 → 余弦相似度检索 → LLM 生成回答
```

### 多 Agent 协作

```
研究员 Agent: 搜索 3 次 → 提取关键信息 → 输出研究摘要
写手 Agent:   接收摘要 → 组织语言 → 生成专业报告
```

## 📝 待办

- [ ] 接入真实 Embedding 模型（sentence-transformers）
- [ ] Docker 部署
- [ ] 接入微信/飞书机器人

## 👤 作者

独立开发，全栈 + AI 应用方向。

- 微信：Dongzhou526
