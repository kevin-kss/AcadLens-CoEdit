"""协同重构管线的数据契约（Stage 1 抽骨 / Stage 2 重构）。

零幻觉的落点在这里：凡是模型「不知道」的数据、指标与案例，一律表达为
SlotCard 交还作者填写，而不是让模型编一个看起来合理的数值。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.diagnosis import Lang
from app.schemas.terms import TermGuardResult


class SlotKind(StrEnum):
    """填空槽位的类型，决定前端用什么输入控件。"""

    NUMERIC = "numeric"
    DATASET = "dataset"
    METRIC = "metric"
    CITATION = "citation"
    CASE = "case"
    CONDITION = "condition"
    OTHER = "other"


class FactTriple(BaseModel):
    """从原段抽出的核心事实三元组。

    必须可溯源到原文：source_quote 是原段中的原话片段，供前端高亮比对，
    也是人工核验「模型有没有擅自加料」的依据。
    """

    subject: str
    predicate: str
    obj: str = Field(description="宾语。字段名避开 Python 关键字 object 的惯用缩写")
    source_quote: str = Field(default="", description="该事实在原段中的出处原话")


class SlotCard(BaseModel):
    """一处待补充的具体数据，以卡片形式交还作者。"""

    slot_id: str
    kind: SlotKind = SlotKind.OTHER
    prompt: str = Field(description="向作者提出的具体问题，如「请补充具体样本量」")
    rationale: str = Field(default="", description="为什么这里需要数据：空洞表达的原文")
    value: str | None = Field(default=None, description="作者填入的真实值；None 表示跳过")

    @property
    def is_filled(self) -> bool:
        return bool(self.value and self.value.strip())


class DeconstructResult(BaseModel):
    """Stage 1 产出：事实提纲 + 待补充槽位。"""

    paragraph_id: str
    lang: Lang
    facts: list[FactTriple] = Field(default_factory=list)
    slots: list[SlotCard] = Field(default_factory=list)
    locked_terms: list[str] = Field(default_factory=list)
    outline: str = Field(default="", description="供 Stage 2 使用的事实提纲文本")


class DraftStrategy(StrEnum):
    """Stage 2 的生成策略。

    只保留一种（PLAIN）。技术文档 5.2 原设计是三版并列供用户挑选，实测后
    收敛为一版：

    - 「错落版」为了打破句长把同样的信息摊得更长，实质是扩写，对想直接复制
      粘贴的用户是负担；而且实测它的风险分（36.2）反而高于直叙版（22.5），
      排序都反了 —— 并列三版等于把系统该做的判断推给用户。
    - 「盲写提纲」不是成稿，用户要的是能直接用的文本。

    「打破句长均匀」这条要求没有丢，已合并进 PLAIN 的提示词 —— 它是降低
    AI 味的主要手段，不能随错落版一起删掉。

    枚举形态保留（而非改成单个常量）：契约里仍是可扩展的策略位，
    将来若要加回别的写法不必改动 API 形状。
    """

    PLAIN = "plain"


class DraftCandidate(BaseModel):
    """一份候选草稿及其门禁结果。

    rejected 的草稿也一并返回：前端置灰展示比静默丢弃更可诊断，用户能看到
    「模型试过但没通过」而不是以为系统只生成了一版。
    """

    strategy: DraftStrategy
    masked_text: str
    guard_result: TermGuardResult
    risk_before: float = Field(ge=0, le=100)
    risk_after: float = Field(ge=0, le=100)
    retry_count: int = Field(default=0, ge=0)
    reject_reason: str = Field(default="")

    @property
    def risk_delta(self) -> float:
        """负值表示风险下降（改好了）。"""
        return self.risk_after - self.risk_before

    @property
    def accepted(self) -> bool:
        return not self.reject_reason


class RestructureResult(BaseModel):
    paragraph_id: str
    lang: Lang
    risk_before: float = Field(ge=0, le=100)
    candidates: list[DraftCandidate] = Field(default_factory=list)

    @property
    def accepted_candidates(self) -> list[DraftCandidate]:
        return [c for c in self.candidates if c.accepted]
