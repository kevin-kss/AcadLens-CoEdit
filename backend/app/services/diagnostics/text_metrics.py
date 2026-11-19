"""共享的文本长度度量。

Track B 用它算句长变异系数，Track C 用它算命中密度的分母。放在独立模块是
为了让两轨共用同一个「一句话有多长」的定义 —— 两处各写一份，迟早算出
互相矛盾的分数。
"""

from __future__ import annotations

import re

from app.schemas.diagnosis import Lang
from app.services.parser.masker import MASK_TOKEN_RE

# CJK 统一表意文字（含扩展 A 与兼容区）：每个字符计一个单位。
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# 连续的拉丁字母/数字串计一个单位，允许内部的点、连字符、下划线与撇号，
# 这样 "ResNet-50"、"76.3"、"et al." 这类术语不会被拆成多个单位。
_LATIN_RUN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-_'][A-Za-z0-9]+)*")


def effective_length(sent: str, lang: Lang | str = Lang.ZH) -> int:
    """剔除掩码占位符与标点后的有效长度。

    中文按字计、英文按词计，但中英混排必须同时成立：直接用 `\\w` 做中文
    字符统计会把 ASCII 字母、数字与下划线一并计入，于是
    "模型使用Transformer v2。" 得 17 而非按字的 6，常见中英混排论文的句长
    CV 会系统性偏离。

    这里统一为「CJK 字符数 + 拉丁/数字串数」：中文句得到字数，英文句得到
    词数，混排句两者相加，量纲一致，故 lang 参数不改变计数方式，仅保留
    以兼容调用方签名与未来的分语种微调。
    """
    _ = lang
    cleaned = MASK_TOKEN_RE.sub("", sent)
    return len(_CJK_RE.findall(cleaned)) + len(_LATIN_RUN_RE.findall(cleaned))
