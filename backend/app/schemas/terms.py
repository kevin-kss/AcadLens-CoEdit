"""术语预提取与锁定校验的数据契约。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.diagnosis import Lang


class TermSource(StrEnum):
    AUTO_EXTRACTED = "auto_extracted"
    USER_ADDED = "user_added"


class TermCandidate(BaseModel):
    """一个候选专有名词及其打分依据。

    score 之外把 frequency / word_count / nested_in 一并透出，是为了让用户在
    术语白名单面板里看懂「为什么这个词被推荐」—— 与诊断的白盒原则一致。
    """

    term: str
    score: float = Field(ge=0, description="C-Value 得分，越高越可能是独立术语")
    frequency: int = Field(ge=1, description="在全文中的出现次数")
    word_count: int = Field(ge=1, description="构成该短语的词数（中文为词，英文为单词）")
    nested_in: int = Field(
        default=0, ge=0, description="作为子串被更长候选包含的次数，用于 C-Value 折减"
    )
    source: TermSource = TermSource.AUTO_EXTRACTED


class LockedTerm(BaseModel):
    term: str
    source: TermSource = TermSource.USER_ADDED
    is_active: bool = True


class TermGuardViolationType(StrEnum):
    MASK_TOKEN_MISSING = "mask_token_missing"
    MASK_TOKEN_ALTERED = "mask_token_altered"
    LOCKED_TERM_MISSING = "locked_term_missing"


class TermGuardViolation(BaseModel):
    """一条门禁违规。结构化而非拼接字符串，便于直接回灌给 LLM 做修正重试。"""

    violation_type: TermGuardViolationType
    subject: str = Field(description="涉及的掩码 Token 或锁定术语原文")
    expected_count: int = Field(ge=0)
    actual_count: int = Field(ge=0)
    message: str


class TermGuardResult(BaseModel):
    passed: bool
    violations: list[TermGuardViolation] = Field(default_factory=list)

    @property
    def feedback(self) -> str:
        """拼给 LLM 的修正反馈。空字符串表示通过。"""
        if self.passed:
            return ""
        return "\n".join(f"- {v.message}" for v in self.violations)


class TermExtractionResult(BaseModel):
    lang: Lang
    candidates: list[TermCandidate]
    skeleton_source: str = Field(
        default="none", description="pos（spaCy 词性过滤）| ngram（无模型降级）"
    )
