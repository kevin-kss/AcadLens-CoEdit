# AcadLens-CoEdit

学术文本白盒诊断与人机协同精修系统。

对学术文稿做可解释的「AI 味」与低信息密度诊断，并在编辑器内与作者协同改写：诊断结果逐句可追溯，改写过程不捏造事实，缺失的数据与案例以填空槽位交还作者填写。

## 设计要点

- **非正文绝对隔离**：公式、引文角标、图表标题与参考文献在进入任何统计或生成模块前替换为占位符。
- **诊断与生成解耦**：本地诊断引擎零外部 Token 消耗，同时作为生成管线的后置质量门禁。
- **术语硬约束**：学科专有名词由代码级字符串校验锁定，不单纯依赖提示词。
- **弹性降级**：无 GPU 时以「句法 + 词库」双轨运行；挂载 GPU 后经环境变量开启本地小模型概率曲率，构成三轨模式。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 前端 | Next.js 14+ (App Router)、TypeScript、TipTap / ProseMirror、Zustand |
| 后端 | Python 3.11+、FastAPI、Pydantic Settings、SQLAlchemy |
| 存储 | SQLite（MVP / 单机）→ PostgreSQL（SaaS 生产） |
| 模型 | OpenAI 兼容接口 / 本地 Ollama / vLLM 可插拔 |

## 目录结构

```text
.
├── dictionaries/   # 静态规则与词库资源（支持热加载）
├── backend/        # FastAPI 服务端
├── frontend/       # Next.js 工作台
└── doc/            # 技术开发文档
```

完整目录规范与接口契约见 [doc/technical_dev_spec.md](doc/technical_dev_spec.md)。

## 开始开发

```bash
cp .env.example .env    # 填入模型接口地址与密钥

# 后端
cd backend && uv sync && uv run uvicorn app.main:app --reload

# 前端
cd frontend && pnpm install && pnpm dev
```

> 工程脚手架尚未落地，上述命令待 `backend/` 与 `frontend/` 初始化后可用。

## 状态

规划阶段。技术方案已定稿至 `v1.0.0`，正在搭建工程骨架。
