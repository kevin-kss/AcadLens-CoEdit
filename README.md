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

# 后端：诊断内核与单测
cd backend
uv sync                 # 基础依赖
uv sync --extra models  # 追加中英文 spaCy 依存模型（启用 Track B 真实依存骨架）
uv run pytest           # 185 条用例
uv run ruff check . && uv run mypy app tests
```

未装 `models` 时 Track B 自动降级为词性近似骨架，单测与 CI 仍可跑；降级状态经
`SentenceMetric.skeleton_source`（`dep` / `pos`）透出到前端。详见
[backend/README.md](backend/README.md)。

HTTP 接口（`app/main.py`）与前端工作台尚未落地，故暂无 `uvicorn` / `pnpm` 启动命令。

## 状态

M1 已完成（诊断内核）：学术分句器、双向掩码器、Track B（句长变异系数与依存骨架
同构度）、Track C（Aho-Corasick 套话扫描与空洞度）、三轨加权聚合，以及中英词库
（套话各 150+ 条、帽句正则各 30+ 条）。`ruff` 与 `mypy --strict` 已接入 CI。

实测：单段实时重算 2.7 ms（验收 < 60 ms），全篇折算 10,000 字单进程 302 ms
（验收 < 500 ms）。

下一步 M2：`C-Value` 术语预提取与 `validate_locked_terms_preserved` 硬校验门禁，
随后接入 Stage 1 / Stage 2 协同重构管线与 FastAPI 路由层。
