"""C-Value 专有名词预提取。

C-Value（Frantzi et al., 2000）是无监督的术语性打分，用来从文本里挑出
「值得整体保留」的多词短语。核心直觉：一个长短语若总是作为更长短语的一部分
出现，它本身就不是独立术语。

    C-Value(a) = log2|a| * freq(a)                    若 a 从未被嵌套
    C-Value(a) = log2|a| * (freq(a) - nested(a))      否则

其中 |a| 为词数，nested(a) 是 a 作为子串出现在更长候选中的平均次数。
单词候选的 log2(1) = 0 会把分数清零，故对其改用 0.5 的权重（原论文的常见
变体），否则 RMSE、Transformer 这类单词术语永远提不出来。

候选短语优先用 spaCy 词性过滤（连续的 NOUN/PROPN/ADJ 串）；模型缺失时降级为
n-gram + 停用词边界切断，与 Track B 的降级策略一致，降级状态随
TermExtractionResult.skeleton_source 透出。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.config import get_settings
from app.schemas.diagnosis import Lang
from app.schemas.terms import TermCandidate, TermExtractionResult, TermSource
from app.services.diagnostics.lexicon_loader import load_lexicon
from app.services.diagnostics.nlp_provider import get_pipeline
from app.services.parser.masker import strip_mask_tokens

# 构成术语的词性：名词、专有名词、形容词。动词与副词不入候选 ——
# 「显著提升」是套话而非术语，正是 Track C 要标记的对象。
_TERM_POS = frozenset({"NOUN", "PROPN", "ADJ", "X"})

# 候选短语内部允许的字符（降级路径用）：中文、拉丁字母数字、连字符
_ZH_RUN_RE = re.compile(r"[㐀-䶿一-鿿]+")
_LATIN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")

# 纯数字、纯标点、单字符中文不作为术语
_PURE_NUMBER_RE = re.compile(r"^[\d.,%]+$")


def _has_newline(text: str, start: int, end: int) -> bool:
    """两个成分之间是否夹着换行。"""
    gap = text[start:end]
    return "\n" in gap or "\r" in gap


def _is_viable(phrase: str, lang: Lang) -> bool:
    """候选短语的基本门槛，过滤明显不是术语的串。

    跨行短语一律拒绝：原文按偏移切片会把换行与缩进一起带进来，
    「多头自注意力\n机制」与「多头自注意力机制」会被当成两个不同的候选，
    各自分走频次，使真正的术语得分被腰斩。
    """
    stripped = phrase.strip()
    if not stripped or _PURE_NUMBER_RE.match(stripped):
        return False
    # 含换行或连续空白：跨行或跨句切片，不是术语
    if any(c in stripped for c in "\n\r\t") or "  " in stripped:
        return False
    # 首尾不得为连接符（如 "-head self"、"multi-"）
    if stripped[0] in _INTRA_WORD_CONNECTORS or stripped[-1] in _INTRA_WORD_CONNECTORS:
        return False
    # 长度门槛与语种无关：单字中文（「的」「中」）与单个拉丁字母（数学变量
    # x、T）都不是术语。二者放在一起判，避免中英混排段落按 lang 走不同分支
    # 时出现「同一个 'x' 在中文段里算术语、在英文段里不算」的不一致。
    return len(stripped) >= 2


def _word_count(phrase: str, lang: Lang) -> int:
    """短语的词数。中文按 2 字一词近似，英文按空白切分。

    C-Value 只用它取 log2，无需精确分词 —— 量纲一致即可。
    """
    if lang is Lang.EN:
        return max(1, len(phrase.split()))
    cjk_chars = sum(len(run) for run in _ZH_RUN_RE.findall(phrase))
    latin_runs = len(_LATIN_TERM_RE.findall(phrase))
    return max(1, round(cjk_chars / 2) + latin_runs)


# 词内连接符：这些 token 出现在两个术语片段之间时不切断短语。
# spaCy 会把 "multi-head self-attention" 切成 multi / - / head / self / - /
# attention，而 "自注意力机制" 中的「自」被标为 ADP。若按词性硬切，抽出来的
# 只会是 'head self' 这类碎片，而非真正的术语。
_INTRA_WORD_CONNECTORS = frozenset({"-", "_", "/", "自", "式", "型"})


def _candidates_from_pos(text: str, lang: Lang) -> list[str] | None:
    """用 spaCy 词性标注抽连续名词短语。模型不可用时返回 None。

    短语文本一律用**原文字符偏移**切片（text[a:b]）而非把 token 拼接起来：
    拼接要猜连接符（英文用空格会把 self-attention 变成 "self attention"，
    中文用空串会粘掉本有的空格），而原文切片天然保真。
    """
    handle = get_pipeline(lang)
    if not handle.available or handle.nlp is None:
        return None

    lexicon = load_lexicon(lang)
    phrases: list[str] = []
    for doc in handle.nlp.pipe([text], batch_size=1):
        # 每个成分记为 (start, end, is_connector)：连接符不计入词数预算，
        # 否则 "multi-head self-attention" 占 6 个 span，超过 term_max_words
        # 而永远拼不出完整术语。
        run: list[tuple[int, int, bool]] = []
        for token in doc:
            if token.is_space:
                continue
            # 跨行/跨段的串不是术语：先在换行处断开 run，再判断当前 token。
            # 用断开而非在 _is_viable 里丢弃 —— 后者会把该 run 内所有合法的
            # 短子串一并连坐丢掉。
            if run and _has_newline(text, run[-1][1], token.idx):
                phrases.extend(_subphrases(text, run, lang))
                run = []
            is_term = (
                token.pos_ in _TERM_POS
                and not token.is_punct
                and token.text.lower() not in lexicon.stopwords
            )
            if is_term:
                run.append((token.idx, token.idx + len(token.text), False))
                continue
            # 连接符只在短语内部起桥接作用，不能作为短语的开头；
            # 且其后必须紧跟（无空白）另一个成分，否则是行末破折号之类的噪声。
            if run and token.text in _INTRA_WORD_CONNECTORS and not token.whitespace_:
                run.append((token.idx, token.idx + len(token.text), True))
                continue
            if run:
                phrases.extend(_subphrases(text, run, lang))
                run = []
        if run:
            phrases.extend(_subphrases(text, run, lang))
    return phrases


def _subphrases(
    text: str,
    spans: list[tuple[int, int, bool]],
    lang: Lang,
) -> list[str]:
    """一段连续名词串的所有子短语（含自身），按原文偏移切片。

    C-Value 需要同时看到「自注意力机制」与「注意力机制」才能判断前者是否
    为独立术语，因此必须枚举子串而非只取最长串。

    spans 的第三位标记该成分是否为连接符；连接符不计入 term_max_words 预算，
    也不能作为子短语的起点或终点。
    """
    max_words = get_settings().term_max_words
    out: list[str] = []
    n = len(spans)
    for i in range(n):
        if spans[i][2]:  # 连接符不作起点
            continue
        words = 0
        for j in range(i, n):
            if not spans[j][2]:
                words += 1
            if words > max_words:
                break
            if spans[j][2]:  # 连接符不作终点
                continue
            phrase = text[spans[i][0] : spans[j][1]]
            if _is_viable(phrase, lang):
                out.append(phrase)
    return out


def _ngram_slices(
    text: str,
    spans: list[tuple[int, int]],
    lang: Lang,
    max_words: int,
) -> list[str]:
    """降级路径的 n-gram 切片，同样按原文偏移取，保证保真。"""
    out: list[str] = []
    n = len(spans)
    for i in range(n):
        for j in range(i + 1, min(n, i + max_words) + 1):
            phrase = text[spans[i][0] : spans[j - 1][1]]
            if _is_viable(phrase, lang):
                out.append(phrase)
    return out


def _candidates_from_ngram(text: str, lang: Lang) -> list[str]:
    """无模型降级：按停用词与标点切段，再在段内取 n-gram。"""
    lexicon = load_lexicon(lang)
    settings = get_settings()
    phrases: list[str] = []

    if lang is Lang.EN:
        max_words = settings.term_max_words
        # 以原文偏移滑窗，保证切片保真（连字符术语不被空格拼接破坏）
        spans = [
            (m.start(), m.end())
            for m in re.finditer(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*", text)
            if m.group(0).lower() not in lexicon.stopwords
        ]
        # 相邻且中间只隔空格的词可组成短语；隔了标点则断开
        run: list[tuple[int, int]] = []
        for span in spans:
            gap = text[run[-1][1] : span[0]] if run else None
            if gap is not None and gap.strip() == "" and "\n" not in gap and "\r" not in gap:
                run.append(span)
                continue
            if run:
                phrases.extend(_ngram_slices(text, run, lang, max_words))
            run = [span]
        if run:
            phrases.extend(_ngram_slices(text, run, lang, max_words))
        return phrases

    # 中文：先按标点与拉丁串切块，再在中文块内滑窗取 2~max 字
    for block in re.split(r"[^㐀-䶿一-鿿A-Za-z0-9\-_]+", text):
        if not block:
            continue
        for latin in _LATIN_TERM_RE.findall(block):
            if _is_viable(latin, lang):
                phrases.append(latin)
        for cjk_run in _ZH_RUN_RE.findall(block):
            max_chars = settings.term_max_words * 2
            for start in range(len(cjk_run)):
                for end in range(start + 2, min(len(cjk_run), start + max_chars) + 1):
                    phrase = cjk_run[start:end]
                    if phrase in lexicon.stopwords:
                        continue
                    if _is_viable(phrase, lang):
                        phrases.append(phrase)
    return phrases


def _nested_counts(freqs: Counter[str]) -> dict[str, int]:
    """统计每个候选作为子串出现在更长候选中的频次之和。

    注意用 `in` 做子串判断而非分词级包含：中文无词边界，「自注意力」确实是
    「多头自注意力机制」的子串，这正是 C-Value 要折减的情形。
    """
    terms = sorted(freqs, key=len, reverse=True)
    nested: dict[str, int] = dict.fromkeys(terms, 0)
    for i, short in enumerate(terms):
        for long in terms[:i]:
            if len(long) > len(short) and short in long:
                nested[short] += freqs[long]
    return nested


def extract_terms(
    text: str,
    lang: Lang | str = Lang.ZH,
    top_k: int | None = None,
) -> TermExtractionResult:
    """从（已掩码的）全文中提取候选专有名词，按 C-Value 降序返回。

    text 应为掩码后的正文：占位符会先被剔除，避免 `{{CITE_1}}` 里的
    CITE 被当成术语。
    """
    lang_enum = Lang(lang)
    settings = get_settings()
    limit = top_k if top_k is not None else settings.term_top_k
    cleaned = strip_mask_tokens(text)

    pos_phrases = _candidates_from_pos(cleaned, lang_enum)
    if pos_phrases is not None:
        phrases, source = pos_phrases, "pos"
    else:
        phrases, source = _candidates_from_ngram(cleaned, lang_enum), "ngram"

    freqs = Counter(phrases)
    # 低频候选不进 C-Value：噪声多且 nested 统计是 O(n²)，先截断可观地省时
    freqs = Counter({t: c for t, c in freqs.items() if c >= settings.term_min_frequency})
    if not freqs:
        return TermExtractionResult(lang=lang_enum, candidates=[], skeleton_source=source)

    nested = _nested_counts(freqs)
    candidates: list[TermCandidate] = []
    for term, freq in freqs.items():
        words = _word_count(term, lang_enum)
        # 单词候选的 log2(1)=0 会把分数清零，故改用 0.5 权重，
        # 否则 RMSE、Transformer 这类单词术语永远提不出来。
        weight = math.log2(words) if words > 1 else 0.5
        nested_count = nested.get(term, 0)
        effective = freq - nested_count / max(1, words) if nested_count else freq
        score = weight * max(0.0, effective)
        if score < settings.term_min_score:
            continue
        candidates.append(
            TermCandidate(
                term=term,
                score=score,
                frequency=freq,
                word_count=words,
                nested_in=nested_count,
                source=TermSource.AUTO_EXTRACTED,
            )
        )

    # 同分时长短语优先：它携带的信息更具体
    candidates.sort(key=lambda c: (-c.score, -len(c.term), c.term))
    return TermExtractionResult(
        lang=lang_enum,
        candidates=candidates[:limit],
        skeleton_source=source,
    )
