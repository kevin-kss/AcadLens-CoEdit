"""协同重构端点：Stage 1 抽骨与 Stage 2 重构。

这两个端点是全系统唯一消耗外部 Token 的地方。模型不可用时返回 503 并带上
原因，不伪造空结果 —— 前端据此提示「模型未配置」，而不是显示一个空的槽位
列表让用户以为本段无需补充数据。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.api import DeconstructRequest, RestructureRequest
from app.schemas.assist import DeconstructResult, RestructureResult
from app.services.document_store import DocumentNotFoundError, DocumentStore, get_store
from app.services.llm.client import ChatClient, LLMError, build_default_client
from app.services.llm.pipeline import deconstruct_paragraph, restructure_paragraph

router = APIRouter(prefix="/assist", tags=["assist"])


def get_llm_client() -> ChatClient:
    """FastAPI 依赖注入入口。测试覆盖它即可完全脱离网络。"""
    return build_default_client()


@router.post("/deconstruct", response_model=DeconstructResult)
def deconstruct(
    payload: DeconstructRequest,
    document_id: str,
    store: DocumentStore = Depends(get_store),
    client: ChatClient = Depends(get_llm_client),
) -> DeconstructResult:
    """Stage 1：低温提取事实三元组，并为空洞表达生成填空槽位。"""
    try:
        document = store.get(document_id)
        paragraph = document.paragraph(payload.paragraph_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    try:
        return deconstruct_paragraph(
            paragraph_id=paragraph.paragraph_id,
            masked_text=paragraph.masked_text,
            client=client,
            lang=document.lang,
            locked_terms=document.locked_terms,
        )
    except LLMError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"模型服务不可用：{exc}",
        ) from exc


@router.post("/restructure", response_model=RestructureResult)
def restructure(
    payload: RestructureRequest,
    document_id: str,
    store: DocumentStore = Depends(get_store),
    client: ChatClient = Depends(get_llm_client),
) -> RestructureResult:
    """Stage 2：按策略生成候选草稿，逐份过术语门禁并复测风险分。

    门禁未通过的候选也会返回（带 reject_reason），由前端置灰展示。
    """
    try:
        document = store.get(document_id)
        paragraph = document.paragraph(payload.paragraph_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    outline = DeconstructResult(
        paragraph_id=paragraph.paragraph_id,
        lang=document.lang,
        facts=[],
        slots=payload.slots,
        locked_terms=document.locked_terms,
        outline="\n".join(f"- {s.prompt}: {s.value}" for s in payload.slots if s.is_filled),
    )

    try:
        return restructure_paragraph(
            original_masked=paragraph.masked_text,
            outline=outline,
            client=client,
            strategies=payload.strategies,
        )
    except LLMError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"模型服务不可用：{exc}",
        ) from exc
