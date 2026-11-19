"""FastAPI 入口。

诊断与重构是 CPU 密集的同步计算，端点一律定义为 `def` 而非 `async def`，
由 FastAPI 放进线程池执行 —— 写成异步会在 GIL 下阻塞事件循环，让并发请求
互相排队（见项目记忆）。

CORS 放开 localhost 是因为前端 dev server 跑在 3000 端口、后端跑在 8000，
跨端口请求必须显式允许。生产部署应改为具体域名。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.assist import router as assist_router
from app.api.v1.documents import router as documents_router

app = FastAPI(
    title="AcadLens-CoEdit",
    version="0.1.0",
    summary="学术文本白盒诊断与人机协同精修",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/api/v1")
app.include_router(assist_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
