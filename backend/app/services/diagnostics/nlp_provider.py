"""spaCy 管线的惰性加载与降级封装。

Track B 的依存骨架需要真实的 parser。模型体积大且加载慢（约 1~3s），因此：
1. 按语种惰性加载，进程内只加载一次；
2. 禁用与依存无关的组件（ner / lemmatizer / attribute_ruler），缩短加载与推理时间；
3. 模型缺失时降级为**词性近似骨架**而非崩溃 —— CI 不装模型也要能跑单测。

降级状态通过 SentenceMetric.skeleton_source 透出到前端（dep / pos / none），
让「这个分数由什么算出来」始终可追溯。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import get_settings
from app.schemas.diagnosis import Lang

if TYPE_CHECKING:
    from spacy.language import Language

logger = logging.getLogger(__name__)

# 依存分析无需这些组件，禁用以加速。
# 注意 attribute_ruler 必须保留：它负责把 tag_ 映射为通用词性 pos_，
# 禁用后 Token.pos_ 恒为空串，骨架签名会退化成纯依存标签序列。
_DISABLED_PIPES = ["ner", "lemmatizer", "textcat"]


@dataclass(frozen=True)
class PipelineHandle:
    nlp: Language | None
    model_name: str
    available: bool
    reason: str = ""


_PIPELINES: dict[Lang, PipelineHandle] = {}


def _model_name(lang: Lang) -> str:
    settings = get_settings()
    return settings.spacy_model_zh if lang is Lang.ZH else settings.spacy_model_en


def get_pipeline(lang: Lang | str) -> PipelineHandle:
    """取得某语种的 spaCy 管线；不可用时返回 available=False 的句柄。"""
    lang_enum = Lang(lang)
    cached = _PIPELINES.get(lang_enum)
    if cached is not None:
        return cached

    name = _model_name(lang_enum)
    settings = get_settings()
    try:
        import spacy

        nlp = spacy.load(name, disable=_DISABLED_PIPES)
        handle = PipelineHandle(nlp=nlp, model_name=name, available=True)
    except Exception as exc:  # 任何加载失败都应降级而非中断诊断
        reason = f"{type(exc).__name__}: {exc}"
        if settings.spacy_require_model:
            raise
        logger.warning(
            "spaCy 模型 %s 不可用，Track B 降级为词性近似骨架（%s）。"
            "安装方式：uv sync --extra models",
            name,
            reason,
        )
        handle = PipelineHandle(nlp=None, model_name=name, available=False, reason=reason)

    _PIPELINES[lang_enum] = handle
    return handle


def reset_pipelines() -> None:
    """测试用：清空已加载管线。"""
    _PIPELINES.clear()
