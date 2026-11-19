"""文档解析层的数据契约：分句结果与掩码结果。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.diagnosis import Lang


class SentenceSpan(BaseModel):
    """一个句子及其在源文本中的精确位置。

    保留 start_char / end_char 是为了让 Track C 命中的词语偏移能从「句内坐标」
    无损换算回「段落坐标」—— 前端下划线定位直接依赖这一点。
    """

    index: int = Field(ge=0)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @property
    def length(self) -> int:
        return self.end_char - self.start_char


class MaskRegistry(BaseModel):
    """掩码映射表：{{CITE_1}} -> '[12]'。随文档持久化，导出时据此还原。"""

    mapping: dict[str, str] = Field(default_factory=dict)
    counter: int = 0

    def register(self, kind: str, raw: str) -> str:
        self.counter += 1
        token = f"{{{{{kind}_{self.counter}}}}}"
        self.mapping[token] = raw
        return token

    @property
    def tokens(self) -> list[str]:
        return list(self.mapping)


class MaskedText(BaseModel):
    masked: str
    registry: MaskRegistry


class ParsedParagraph(BaseModel):
    paragraph_id: str
    order_index: int = Field(ge=0)
    section_type: str = "body"
    masked_text: str
    sentences: list[SentenceSpan]
    lang: Lang
