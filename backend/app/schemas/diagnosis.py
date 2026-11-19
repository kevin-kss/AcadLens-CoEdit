"""诊断引擎的内部与对外数据契约。

跨层传递一律用这里的 Pydantic 模型，不传裸 dict —— 诊断结果会直接驱动前端
ProseMirror 的装饰层坐标，字段错位的代价比多写几行类型声明高得多。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Lang(StrEnum):
    ZH = "zh"
    EN = "en"


class EngineMode(StrEnum):
    DUAL_TRACK = "dual_track"
    TRI_TRACK = "tri_track"


class IssueType(StrEnum):
    AI_CONNECTOR = "ai_connector"
    HAT_BOOT_CLICHE = "hat_boot_cliche"
    LOW_INFO_DENSITY = "low_info_density"
    UNIFORM_LENGTH = "uniform_length"
    SYNTACTIC_PARALLEL = "syntactic_parallel"
    HIGH_PROB_CURVATURE = "high_curvature"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_score(cls, score: float) -> RiskLevel:
        if score > 65:
            return cls.HIGH
        if score >= 35:
            return cls.MEDIUM
        return cls.LOW


class TextSpanIssue(BaseModel):
    """一处可定位的问题。

    偏移量以 Python str 索引（即 Unicode 码点）为准，相对于**带掩码的段落文本**。
    前端换算到 ProseMirror 绝对位置前需先还原掩码长度差，见 frontend 的坐标映射层。
    """

    issue_type: IssueType
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    matched_text: str
    message: str

    def overlaps(self, other: TextSpanIssue) -> bool:
        return self.start_char < other.end_char and other.start_char < self.end_char


class SentenceMetric(BaseModel):
    sent_index: int = Field(ge=0)
    text: str
    start_char: int = Field(ge=0, description="该句在段落掩码文本中的起始偏移")
    clean_char_len: int = Field(ge=0, description="剔除掩码 Token 与标点后的真实字/词数")
    ppl: float | None = None
    curvature: float | None = None
    dep_skeleton: str = ""
    skeleton_source: str = Field(default="none", description="dep | pos | none")
    fluff_ratio: float = Field(default=0.0, ge=0)
    concrete_anchors: int = Field(default=0, ge=0)
    sentence_risk: float = Field(default=0.0, ge=0, le=100)


class TrackScores(BaseModel):
    """各轨道归一化到 [0, 100] 的子分数，用于白盒展示「分从哪来」。"""

    syntactic: float = Field(ge=0, le=100)
    lexicon: float = Field(ge=0, le=100)
    curvature: float | None = Field(default=None, ge=0, le=100)


class ParagraphDiagnosis(BaseModel):
    paragraph_id: str
    lang: Lang
    mode: EngineMode
    burstiness_cv: float = Field(ge=0)
    track_scores: TrackScores
    overall_risk: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    sentences: list[SentenceMetric]
    span_issues: list[TextSpanIssue]


class DocumentRiskSummary(BaseModel):
    paragraph_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)
    mean_risk: float = Field(ge=0, le=100)
    high_risk_paragraph_ids: list[str]
    issue_counts: dict[IssueType, int]
