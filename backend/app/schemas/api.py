"""API 层的请求与响应契约。

诊断与术语的返回值直接复用内部 schema（ParagraphDiagnosis、TermCandidate 等），
不另起一套 —— 两套 schema 迟早会字段漂移，而前端坐标定位直接依赖这些字段。
这里只定义「请求」与「文档级」的结构。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.assist import DraftStrategy, SlotCard
from app.schemas.diagnosis import Lang, ParagraphDiagnosis, RiskLevel
from app.schemas.terms import TermCandidate


class ParagraphView(BaseModel):
    """前端编辑器里的一段。

    masked_text 是编辑器实际编辑的内容（含 `{{CITE_1}}` 占位符）；
    plain_text 是还原后的可读文本，仅用于展示，不参与诊断。
    """

    paragraph_id: str
    order_index: int = Field(ge=0)
    section_type: str
    masked_text: str
    plain_text: str
    diagnosis: ParagraphDiagnosis


class DocumentSummary(BaseModel):
    paragraph_count: int = Field(ge=0)
    mean_risk: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    high_risk_paragraph_ids: list[str]


class DocumentResponse(BaseModel):
    document_id: str
    lang: Lang
    paragraphs: list[ParagraphView]
    suggested_terms: list[TermCandidate]
    summary: DocumentSummary


class CreateDocumentRequest(BaseModel):
    text: str = Field(min_length=1, description="纯文本稿件，按空行分段")
    lang: Lang = Lang.ZH


class UpdateParagraphRequest(BaseModel):
    """编辑器实时重算：前端回传当前段的带掩码文本。"""

    masked_text: str


class SetTermsRequest(BaseModel):
    terms: list[str] = Field(description="用户确认锁定的术语列表，整体替换")


class DeconstructRequest(BaseModel):
    paragraph_id: str


class FilledSlot(BaseModel):
    slot_id: str
    value: str | None = None


class RestructureRequest(BaseModel):
    paragraph_id: str
    slots: list[SlotCard] = Field(description="Stage 1 返回的槽位，含作者填写的值")
    strategies: list[DraftStrategy] | None = None
