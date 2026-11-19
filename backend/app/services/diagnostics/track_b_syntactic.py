"""轨道 B：句长突发性（Burstiness）与句法骨架同构度。

两个信号：
1. **句长变异系数 CV = σ/μ**。人类学术写作通常 > 0.30，典型 AI 生成 < 0.18 ——
   模型倾向于输出长度均匀的句子。
2. **依存骨架同构度**。取每句 ROOT 的直接子节点依存标签序列作为签名
   （如 "advmod-nsubj-obj-conj"），相邻句签名相似度 >= 阈值即判为排比/句式雷同。

骨架优先用 spaCy 真实依存；模型缺失时降级为词性序列近似，降级状态随
SentenceMetric.skeleton_source 透出，保证白盒可追溯。
"""

from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

from app.config import get_settings
from app.schemas.diagnosis import IssueType, Lang, TextSpanIssue
from app.schemas.document import SentenceSpan
from app.services.diagnostics.nlp_provider import get_pipeline
from app.services.parser.masker import MASK_TOKEN_RE

# 计算有效长度时剔除的标点
_PUNCT_RE_ZH = r"[^\w一-鿿]"


def effective_length(sent: str, lang: Lang | str = Lang.ZH) -> int:
    """剔除掩码占位符与标点后的有效长度。

    中文按字计，英文按词计 —— 两者的「一句话有多长」量纲不同，混用会让
    中英混排段落的 CV 完全失真。
    """
    import re

    cleaned = MASK_TOKEN_RE.sub("", sent).strip()
    if Lang(lang) is Lang.EN:
        return len([w for w in cleaned.split() if any(c.isalnum() for c in w)])
    return len(re.sub(_PUNCT_RE_ZH, "", cleaned))


def compute_burstiness_cv(
    sentences: list[str],
    lang: Lang | str = Lang.ZH,
) -> float:
    """句长变异系数 CV = std / mean。

    短段落（< min_sentences_for_burstiness 句）返回一个中性值，不做低波动惩罚：
    两句话的长度方差没有统计意义，罚分只会制造噪音。
    """
    settings = get_settings()
    if len(sentences) < settings.min_sentences_for_burstiness:
        return 0.5
    lengths = np.array(
        [max(1, effective_length(s, lang)) for s in sentences],
        dtype=float,
    )
    mean = float(np.mean(lengths))
    if mean <= 0:
        return 0.5
    return float(np.std(lengths) / mean)


def _dep_skeleton(doc: object) -> str:
    """ROOT 直接子节点的依存标签序列，按在句中的出现顺序。

    只取顶层（而非整棵树）是刻意的：顶层骨架反映主干句式，对修饰语的增删
    不敏感，正适合判断「两句是不是同一个模子」。
    """
    tokens = list(doc)  # type: ignore[call-overload]
    if not tokens:
        return ""
    # 必须按 token 下标比较，不能用 `is`：spaCy 的 Token 是每次访问重建的轻量
    # 代理对象，`t.head is t` 恒为 False，会让所有骨架静默变成空串。
    root = next((t for t in tokens if t.head.i == t.i), None)
    if root is None:
        return ""
    children = [t for t in tokens if t.head.i == root.i and t.i != root.i]
    labels = [t.dep_ for t in sorted(children, key=lambda t: t.i)]
    return "-".join([root.pos_, *labels]) if labels else root.pos_


def _pos_skeleton_fallback(sent: str, lang: Lang) -> str:
    """无 spaCy 模型时的近似骨架。

    英文用功能词序列，中文用虚词/标点序列。精度明显低于真依存，仅用于保证
    CI 与无模型环境可运行；阈值不同，故 skeleton_source 必须一并透出。
    """
    import re

    cleaned = MASK_TOKEN_RE.sub("", sent)
    if lang is Lang.EN:
        function_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "from",
            "and",
            "or",
            "but",
            "that",
            "which",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "has",
            "have",
            "had",
            "can",
            "could",
            "may",
            "will",
            "would",
            "not",
            "this",
            "these",
            "it",
            "we",
            "our",
        }
        words = re.findall(r"[A-Za-z']+", cleaned.lower())
        return "-".join(w if w in function_words else "X" for w in words)

    particles = set("的了着过地得之与和及或等在是为把被将从向对于以由并且而则就")
    return "-".join(c if c in particles or not c.isalnum() else "X" for c in cleaned)


def build_dep_skeletons(
    sentences: list[str],
    lang: Lang | str = Lang.ZH,
) -> tuple[list[str], str]:
    """批量构造句法骨架。

    返回 (骨架列表, 来源标记)，来源为 "dep" 或 "pos"。
    走 nlp.pipe 批处理：逐句调用 nlp() 在整篇诊断上会慢数倍。
    """
    lang_enum = Lang(lang)
    if not sentences:
        return [], "none"

    handle = get_pipeline(lang_enum)
    if handle.available and handle.nlp is not None:
        docs = handle.nlp.pipe(sentences, batch_size=64)
        return [_dep_skeleton(d) for d in docs], "dep"

    return [_pos_skeleton_fallback(s, lang_enum) for s in sentences], "pos"


def detect_syntactic_parallelism(
    dep_skeletons: list[str],
    threshold: float | None = None,
) -> list[int]:
    """返回与相邻句结构高度雷同的句子下标。

    只比相邻句：跨句比对会把「同一段里两处恰好同构」误判为排比，而排比的
    阅读观感恰恰来自连续出现。
    """
    if threshold is None:
        threshold = get_settings().parallel_similarity_threshold

    parallel: set[int] = set()
    for i in range(1, len(dep_skeletons)):
        prev, cur = dep_skeletons[i - 1], dep_skeletons[i]
        if not prev or not cur:
            continue
        # 极短骨架（单 token）天然容易同构，不参与判定
        if min(prev.count("-"), cur.count("-")) < 2:
            continue
        if SequenceMatcher(None, prev, cur).ratio() >= threshold:
            parallel.update((i - 1, i))
    return sorted(parallel)


def _normalize_burstiness(cv: float) -> float:
    """CV -> [0,100] 风险分，分段线性。

    CV <= ai_ceiling(0.18) 记满分风险；>= human_floor(0.30) 记 0；中间线性插值。
    """
    settings = get_settings()
    low, high = settings.burstiness_ai_ceiling, settings.burstiness_human_floor
    if cv <= low:
        return 100.0
    if cv >= high:
        return 0.0
    return float(100.0 * (high - cv) / (high - low))


def diagnose_track_b(
    sentences: list[SentenceSpan],
    lang: Lang | str = Lang.ZH,
) -> tuple[float, float, list[str], str, list[TextSpanIssue]]:
    """Track B 主入口。

    返回 (syntactic_score, burstiness_cv, skeletons, skeleton_source, issues)。
    """
    lang_enum = Lang(lang)
    texts = [s.text for s in sentences]
    cv = compute_burstiness_cv(texts, lang_enum)
    skeletons, source = build_dep_skeletons(texts, lang_enum)
    parallel_idx = detect_syntactic_parallelism(skeletons)

    issues: list[TextSpanIssue] = []
    settings = get_settings()

    if len(texts) >= settings.min_sentences_for_burstiness and cv <= settings.burstiness_ai_ceiling:
        # 句长均质是段落级属性，锚定到最长句以便前端定位
        longest = max(sentences, key=lambda s: effective_length(s.text, lang_enum))
        issues.append(
            TextSpanIssue(
                issue_type=IssueType.UNIFORM_LENGTH,
                start_char=longest.start_char,
                end_char=longest.end_char,
                matched_text=longest.text,
                message=(
                    f"本段 {len(texts)} 句的句长变异系数仅 {cv:.2f}"
                    f"（人类学术写作通常 > {settings.burstiness_human_floor:.2f}），"
                    "句子长度过于均匀。建议插入一句短判断句打破节奏。"
                ),
            )
        )

    for idx in parallel_idx:
        span = sentences[idx]
        issues.append(
            TextSpanIssue(
                issue_type=IssueType.SYNTACTIC_PARALLEL,
                start_char=span.start_char,
                end_char=span.end_char,
                matched_text=span.text,
                message=(
                    "与相邻句的句法骨架高度同构"
                    f"（骨架 {skeletons[idx] or 'N/A'}，来源 {source}）。"
                    "连续同构句式是生成文本的典型特征，建议改写其中一句的主干结构。"
                ),
            )
        )

    parallel_ratio = len(parallel_idx) / len(texts) if texts else 0.0
    burst_score = _normalize_burstiness(cv)
    # 句长均质与句式雷同各占一半：两者独立出现都足以构成「读起来像机器写的」
    syntactic_score = min(100.0, 0.5 * burst_score + 0.5 * (100.0 * parallel_ratio))

    return syntactic_score, cv, skeletons, source, issues
