"""API 层的业务编排：把解析、掩码、诊断、术语提取串成一次文档创建。

路由层只做参数校验与错误码映射，所有「先做什么后做什么」的顺序都在这里 ——
否则每个端点都会各自写一遍掩码-分句-诊断的流程，顺序一处写错就会让
公式计入句长、诊断分数失真。
"""

from __future__ import annotations

from app.schemas.api import (
    DocumentResponse,
    DocumentSummary,
    ParagraphView,
)
from app.schemas.diagnosis import Lang, RiskLevel
from app.schemas.document import ParsedParagraph
from app.services.diagnostics.engine import diagnose_document
from app.services.document_store import DocumentStore, StoredDocument
from app.services.guardrails.term_extractor import extract_terms
from app.services.parser.masker import mask_academic_text, unmask_academic_text
from app.services.parser.splitter import split_paragraphs, split_sentences


def _build_paragraphs(
    text: str,
    lang: Lang,
) -> tuple[list[ParsedParagraph], object]:
    """分段、逐段掩码、分句。

    各段共享同一个 MaskRegistry，保证 `{{CITE_1}}` 在全文范围内编号唯一，
    还原时不会出现两个段落争用同一个占位符的情况。
    """
    from app.schemas.document import MaskRegistry

    registry = MaskRegistry()
    paragraphs: list[ParsedParagraph] = []
    for index, raw in enumerate(split_paragraphs(text)):
        masked = mask_academic_text(raw, registry)
        paragraphs.append(
            ParsedParagraph(
                paragraph_id=f"p{index}",
                order_index=index,
                section_type="body",
                masked_text=masked.masked,
                sentences=split_sentences(masked.masked, lang),
                lang=lang,
            )
        )
    return paragraphs, registry


def create_document(store: DocumentStore, text: str, lang: Lang) -> DocumentResponse:
    """导入纯文本：掩码 -> 分句 -> 全篇诊断 -> 术语预提取 -> 入库。"""
    paragraphs, registry = _build_paragraphs(text, lang)
    document = store.create(lang, paragraphs, registry)  # type: ignore[arg-type]
    return document_response(document)


def document_response(document: StoredDocument) -> DocumentResponse:
    """把存储中的文档组装成前端需要的视图（含最新诊断）。"""
    diagnoses, summary = diagnose_document(document.paragraphs)

    views: list[ParagraphView] = []
    for para, diagnosis in zip(document.paragraphs, diagnoses, strict=True):
        views.append(
            ParagraphView(
                paragraph_id=para.paragraph_id,
                order_index=para.order_index,
                section_type=para.section_type,
                masked_text=para.masked_text,
                plain_text=unmask_academic_text(para.masked_text, document.registry),
                diagnosis=diagnosis,
            )
        )

    # 术语从全文提取：单段频次太低，C-Value 的嵌套统计需要全文语境
    full_text = "\n".join(p.masked_text for p in document.paragraphs)
    suggested = extract_terms(full_text, document.lang).candidates

    return DocumentResponse(
        document_id=document.document_id,
        lang=document.lang,
        paragraphs=views,
        suggested_terms=suggested,
        summary=DocumentSummary(
            paragraph_count=summary.paragraph_count,
            mean_risk=summary.mean_risk,
            risk_level=RiskLevel.from_score(summary.mean_risk),
            high_risk_paragraph_ids=summary.high_risk_paragraph_ids,
        ),
    )


def update_paragraph(
    document: StoredDocument,
    paragraph_id: str,
    masked_text: str,
) -> ParagraphView:
    """编辑器实时重算：用前端回传的掩码文本替换该段并重新诊断。

    只重算这一段，不重跑全文 —— 这是满足 800ms 防抖下 60ms 响应预算的前提。
    分句在这里重做，因为用户编辑会改变句子边界。
    """
    from app.services.diagnostics.engine import diagnose_paragraph

    para = document.paragraph(paragraph_id)
    para.masked_text = masked_text
    para.sentences = split_sentences(masked_text, document.lang)

    diagnosis = diagnose_paragraph(
        paragraph_id=paragraph_id,
        masked_text=masked_text,
        lang=document.lang,
        sentences=para.sentences,
    )
    return ParagraphView(
        paragraph_id=paragraph_id,
        order_index=para.order_index,
        section_type=para.section_type,
        masked_text=masked_text,
        plain_text=unmask_academic_text(masked_text, document.registry),
        diagnosis=diagnosis,
    )
