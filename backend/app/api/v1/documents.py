"""文档相关端点：导入、读取、实时重算、锁定术语、导出。

这些端点是 **CPU 密集的同步函数**，刻意不写成 `async def` —— 诊断在 GIL 下
会阻塞事件循环，让并发请求互相排队。定义为 `def` 时 FastAPI 自动放进线程池，
是这里想要的行为（见项目记忆：并发应走多进程，异步帮不上忙）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.schemas.api import (
    CreateDocumentRequest,
    DocumentResponse,
    ParagraphView,
    SetTermsRequest,
    UpdateParagraphRequest,
)
from app.services.document_service import (
    create_document,
    document_response,
    update_paragraph,
)
from app.services.document_store import (
    DocumentNotFoundError,
    DocumentStore,
    StoredDocument,
    get_store,
)
from app.services.parser.masker import unmask_academic_text

router = APIRouter(prefix="/documents", tags=["documents"])


def _fetch(store: DocumentStore, document_id: str) -> StoredDocument:
    try:
        return store.get(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/parse", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def parse_document(
    payload: CreateDocumentRequest,
    store: DocumentStore = Depends(get_store),
) -> DocumentResponse:
    """导入纯文本稿件，返回全篇诊断与候选术语。

    .docx 上传待 M4 的 docx_handler 落地后以独立端点接入；此处只收纯文本，
    避免先用一个「能传但不保样式」的半成品端点误导调用方。
    """
    return create_document(store, payload.text, payload.lang)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    store: DocumentStore = Depends(get_store),
) -> DocumentResponse:
    return document_response(_fetch(store, document_id))


@router.put("/{document_id}/paragraphs/{paragraph_id}", response_model=ParagraphView)
def update_paragraph_endpoint(
    document_id: str,
    paragraph_id: str,
    payload: UpdateParagraphRequest,
    store: DocumentStore = Depends(get_store),
) -> ParagraphView:
    """编辑器实时重算单段。前端做 800ms 防抖后调用。"""
    document = _fetch(store, document_id)
    try:
        return update_paragraph(document, paragraph_id, payload.masked_text)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.put("/{document_id}/terms", response_model=DocumentResponse)
def set_locked_terms(
    document_id: str,
    payload: SetTermsRequest,
    store: DocumentStore = Depends(get_store),
) -> DocumentResponse:
    """整体替换锁定术语表。术语会在 Stage 1 / Stage 2 与门禁中生效。"""
    _fetch(store, document_id)
    document = store.set_locked_terms(document_id, payload.terms)
    return document_response(document)


@router.get("/{document_id}/export", response_class=PlainTextResponse)
def export_document(
    document_id: str,
    store: DocumentStore = Depends(get_store),
) -> str:
    """导出纯文本：还原所有掩码占位符。

    .docx 导出（保留原样式 + 修改审计表）待 M4。这里先保证无损还原这条
    硬指标在纯文本链路上成立。
    """
    document = _fetch(store, document_id)
    return "\n\n".join(
        unmask_academic_text(p.masked_text, document.registry) for p in document.paragraphs
    )
