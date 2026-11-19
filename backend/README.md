# AcadLens-CoEdit — 服务端

学术文本白盒诊断引擎与人机协同重构管线的 Python 实现。

## 安装

```bash
uv sync                    # 基础依赖（不含 spaCy 模型）
uv sync --extra models     # 追加中英文依存模型，启用 Track B 真实依存骨架
```

未安装 `models` 时 Track B 自动降级为词性近似骨架，单测与 CI 仍可运行；
降级状态通过 `SentenceMetric.skeleton_source`（`dep` / `pos`）透出到前端。

## 开发命令

```bash
uv run pytest              # 单元测试
uv run ruff check .        # lint
uv run ruff format .       # 格式化
uv run mypy app            # strict 类型检查
```

## 模块地图

| 路径 | 职责 |
| --- | --- |
| `app/schemas/` | 跨层 Pydantic 契约（诊断结果、分句与掩码结构） |
| `app/services/parser/splitter.py` | 中英文学术分句器，跳过 `Fig. 1` / `et al.` / `3.14` |
| `app/services/parser/masker.py` | 公式与引文双向无损掩码 |
| `app/services/diagnostics/track_b_syntactic.py` | 句长变异系数 CV 与依存骨架同构度 |
| `app/services/diagnostics/track_c_lexicon.py` | Aho-Corasick 套话扫描与空洞度 |
| `app/services/diagnostics/engine.py` | 三轨调度与加权聚合 |
| `app/services/diagnostics/lexicon_loader.py` | 词库热加载，坏规则启动即报错 |

词库资源位于仓库根目录 `dictionaries/{zh,en}/`，改动后无需重启进程。

## 并发约定

诊断是 CPU 密集的纯函数，**不要写成 `async def`** —— GIL 会让并发诊断请求
互相排队并阻塞事件循环。并发由上层 `ProcessPoolExecutor` 或多 worker 承担；
Track A 启用后模型常驻内存，其 worker 数需单独限制。
