"""轨道 C：Aho-Corasick 套话扫描、帽句规则匹配与空洞度计算。

三个信号：
1. **AI 套话命中**：用 Aho-Corasick 自动机一遍扫完全部词库（中英各 150+ 条），
   复杂度与词库规模无关，是满足 60ms 实时重算预算的关键。
2. **帽句/升华句**：正则规则，按 head / tail / any 位置生效。
3. **空洞度**：FluffScore(s) = Count(AbstractWords) / (1 + 2 * Count(ConcreteAnchors))。
   抽象词 >= 2 且具象锚点为 0 时直接触发 LOW_INFO_DENSITY，为 Stage 1 生成
   数据填空槽位提供依据。

英文需词边界校验：子串匹配会让 "thus" 命中 "enthusiasm"、"so" 命中 "solution"。
中文无词边界概念，直接按子串命中。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import ahocorasick

from app.config import get_settings
from app.schemas.diagnosis import IssueType, Lang, TextSpanIssue
from app.schemas.document import SentenceSpan
from app.services.diagnostics.lexicon_loader import LangLexicon, load_lexicon
from app.services.diagnostics.text_metrics import effective_length

# ---------------------------------------------------------------------------
# 具象锚点（Concrete Anchors）
# ---------------------------------------------------------------------------
# 定义：阿拉伯数字、百分比、掩码 Token（引文/公式）、比较级数值表达。
# 一个句子只要有具象锚点，抽象词就有了支撑，空洞度随之降低。
_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\{\{(?:CITE|MATH|LATEX_CMD|URL)_\d+\}\}"),  # 引文/公式/链接
    re.compile(r"\d+(?:\.\d+)?\s*%"),  # 百分比
    re.compile(r"[a-zA-Z]\s*[=＝<>≤≥]\s*-?\d+(?:\.\d+)?"),  # p < 0.05, n = 120
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:倍|个百分点|pp|ms|s|μs|ns|GB|MB|KB|TB|"
        r"FLOPs|epochs?|iterations?|samples?|images?|tokens?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:提升|提高|下降|降低|减少|增加|改善)了?\s*\d+(?:\.\d+)?"),
    re.compile(
        r"\b(?:Table|Fig(?:ure)?|Eq(?:uation)?|Sec(?:tion)?|Alg(?:orithm)?)\s*\.?\s*\d+",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}\s*年|\b(?:19|20)\d{2}\b"),  # 年份
    re.compile(r"\b\d+(?:\.\d+)?\b"),  # 兜底：任意数字
)


@dataclass(frozen=True)
class _Automaton:
    machine: ahocorasick.Automaton
    needs_word_boundary: bool


_AUTOMATON_CACHE: dict[tuple[Lang, str, int], _Automaton] = {}


def _build_automaton(words: tuple[str, ...], lang: Lang) -> _Automaton:
    machine = ahocorasick.Automaton()
    for word in words:
        key = word.lower() if lang is Lang.EN else word
        # 存原词而非 key：命中后要把原始词面回传给前端提示
        machine.add_word(key, (len(key), word))
    machine.make_automaton()
    return _Automaton(machine=machine, needs_word_boundary=lang is Lang.EN)


def _get_automaton(words: tuple[str, ...], lang: Lang, kind: str) -> _Automaton:
    cache_key = (lang, kind, hash(words))
    cached = _AUTOMATON_CACHE.get(cache_key)
    if cached is None:
        cached = _build_automaton(words, lang)
        _AUTOMATON_CACHE[cache_key] = cached
    return cached


def _lower_preserving_length(text: str) -> str:
    """逐码点小写化，保证输出与输入长度一致。

    直接 text.lower() 在 Unicode 下会改变长度：'İ'(U+0130) 小写为两个码点
    'i'+U+0307。长度一变，自动机给出的命中下标就无法用于切原文 ——
    前端拿到的偏移会整体错位。逐字符转换时，若某字符的小写展开为多码点
    则保留原字符（它不可能是 ASCII 词库条目的一部分，不影响匹配）。
    """
    if text.isascii():
        return text.lower()
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """英文命中需两侧均非字母数字，避免 thus 命中 enthusiasm。"""
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _scan_phrases(
    text: str,
    words: tuple[str, ...],
    lang: Lang,
    kind: str,
) -> list[tuple[int, int, str]]:
    """返回 (start, end, matched_word) 列表，已剔除被更长命中覆盖的短命中。"""
    if not words:
        return []
    auto = _get_automaton(words, lang, kind)
    # 英文需大小写无关匹配，但**不能**用 text.lower()：Unicode 大小写展开会改变
    # 字符串长度（如 "İ".lower() 为两个码点），使命中下标在原文中整体错位，
    # 前端下划线随之画错位置。str.casefold 同样会展开，故改用逐字符的
    # str.lower()，它对每个码点独立转换，长度恒等。
    haystack = _lower_preserving_length(text) if lang is Lang.EN else text

    raw: list[tuple[int, int, str]] = []
    for end_idx, (length, original) in auto.machine.iter(haystack):
        start = end_idx - length + 1
        end = end_idx + 1
        if auto.needs_word_boundary and not _boundary_ok(haystack, start, end):
            continue
        raw.append((start, end, original))

    # 最长优先去重："值得注意的是" 命中后不再单独报 "注意"
    raw.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list[tuple[int, int, str]] = []
    covered_until = -1
    for start, end, word in raw:
        if start < covered_until:
            continue
        kept.append((start, end, word))
        covered_until = end
    return kept


def count_concrete_anchors(sentence: str) -> int:
    """统计句中具象锚点数量（去重叠）。"""
    spans: list[tuple[int, int]] = []
    for pattern in _ANCHOR_PATTERNS:
        spans.extend((m.start(), m.end()) for m in pattern.finditer(sentence))
    if not spans:
        return 0
    spans.sort()
    merged = 1
    cur_end = spans[0][1]
    for start, end in spans[1:]:
        if start >= cur_end:
            merged += 1
            cur_end = end
        else:
            cur_end = max(cur_end, end)
    return merged


def compute_fluff_score(abstract_count: int, anchor_count: int) -> float:
    """FluffScore = AbstractWords / (1 + 2 * ConcreteAnchors)。"""
    return abstract_count / (1.0 + 2.0 * anchor_count)


def _match_hat_boot(
    sentence: SentenceSpan,
    lexicon: LangLexicon,
    is_head: bool,
    is_tail: bool,
) -> list[TextSpanIssue]:
    issues: list[TextSpanIssue] = []
    for rule in lexicon.hat_boot_rules:
        if rule.position == "head" and not is_head:
            continue
        if rule.position == "tail" and not is_tail:
            continue
        match = rule.pattern.search(sentence.text)
        if match is None:
            continue
        issues.append(
            TextSpanIssue(
                issue_type=IssueType.HAT_BOOT_CLICHE,
                start_char=sentence.start_char + match.start(),
                end_char=sentence.start_char + match.end(),
                matched_text=match.group(0),
                message=f"[{rule.rule_id}] {rule.message}",
            )
        )
    return issues


@dataclass
class TrackCSentenceResult:
    fluff_ratio: float
    concrete_anchors: int
    abstract_hits: int
    connector_hits: int
    issues: list[TextSpanIssue]


def diagnose_sentence_track_c(
    sentence: SentenceSpan,
    lexicon: LangLexicon,
    is_head: bool = False,
    is_tail: bool = False,
) -> TrackCSentenceResult:
    """单句的 Track C 诊断。偏移量一律换算为段落坐标。"""
    settings = get_settings()
    lang = lexicon.lang
    text = sentence.text
    base = sentence.start_char
    issues: list[TextSpanIssue] = []

    # 1. AI 套话连接词
    connector_hits = _scan_phrases(text, lexicon.connectors, lang, "connector")
    for start, end, word in connector_hits:
        issues.append(
            TextSpanIssue(
                issue_type=IssueType.AI_CONNECTOR,
                start_char=base + start,
                end_char=base + end,
                matched_text=text[start:end],
                message=(
                    f"显性连接词/套话「{word}」。生成文本依赖此类词标示逻辑，"
                    "人类学术写作更多靠主语指代与因果关系直接连句，可考虑删除。"
                ),
            )
        )

    # 2. 抽象空泛词与具象锚点
    abstract_hits = _scan_phrases(text, lexicon.fluff, lang, "fluff")
    anchors = count_concrete_anchors(text)
    fluff_ratio = compute_fluff_score(len(abstract_hits), anchors)

    if len(abstract_hits) >= settings.fluff_abstract_min_count and anchors == 0:
        phrases = "、".join(w for _, _, w in abstract_hits[:4])
        issues.append(
            TextSpanIssue(
                issue_type=IssueType.LOW_INFO_DENSITY,
                start_char=base,
                end_char=base + len(text),
                matched_text=text,
                message=(
                    f"本句含 {len(abstract_hits)} 个抽象表述（{phrases}）"
                    "但无任何具体数据、引文或公式支撑。"
                    "建议补充数据集、指标数值或实验条件 —— 精修时将生成填空槽位。"
                ),
            )
        )

    # 3. 帽句与升华句
    issues.extend(_match_hat_boot(sentence, lexicon, is_head, is_tail))

    return TrackCSentenceResult(
        fluff_ratio=fluff_ratio,
        concrete_anchors=anchors,
        abstract_hits=len(abstract_hits),
        connector_hits=len(connector_hits),
        issues=issues,
    )


def _normalize_lexicon_score(
    results: list[TrackCSentenceResult],
    sentences: list[SentenceSpan],
    lang: Lang,
) -> float:
    """词库轨道的段落级风险分 -> [0,100]。

    按「每百字命中密度」而非绝对命中数归一化，否则长段落天然吃亏。

    分母必须用 effective_length（剔除掩码占位符与标点）而非原始文本长度：
    `{{CITE_1}}` 这类占位符长 10~12 字符，若计入分母，给段落多加几条引文
    就能稀释密度、凭空降低风险分，而正文一字未改。
    """
    if not results or not sentences:
        return 0.0

    total_len = max(sum(effective_length(s.text, lang) for s in sentences), 1)
    unit = 100.0 if lang is Lang.ZH else 20.0  # 百字 / 二十词

    connector_density = sum(r.connector_hits for r in results) / total_len * unit
    hat_boot_count = sum(
        1 for r in results for i in r.issues if i.issue_type is IssueType.HAT_BOOT_CLICHE
    )
    low_info_count = sum(
        1 for r in results for i in r.issues if i.issue_type is IssueType.LOW_INFO_DENSITY
    )

    # 分段线性：每百字 3 个连接词即接近满分（实测人类学术写作通常 < 1）
    connector_score = min(100.0, connector_density / 3.0 * 100.0)
    # 帽句按句数占比，一段里有一句纯铺垫就已明显
    hat_boot_score = min(100.0, hat_boot_count / len(sentences) * 150.0)
    low_info_score = min(100.0, low_info_count / len(sentences) * 120.0)

    return min(
        100.0,
        0.40 * connector_score + 0.30 * hat_boot_score + 0.30 * low_info_score,
    )


def diagnose_track_c(
    sentences: list[SentenceSpan],
    lang: Lang | str = Lang.ZH,
) -> tuple[float, list[TrackCSentenceResult], list[TextSpanIssue]]:
    """Track C 主入口。返回 (lexicon_score, 每句结果, 全部问题)。"""
    lang_enum = Lang(lang)
    if not sentences:
        return 0.0, [], []

    lexicon = load_lexicon(lang_enum)
    last = len(sentences) - 1
    results = [
        diagnose_sentence_track_c(s, lexicon, is_head=(i == 0), is_tail=(i == last))
        for i, s in enumerate(sentences)
    ]
    issues = [issue for r in results for issue in r.issues]
    score = _normalize_lexicon_score(results, sentences, lang_enum)
    return score, results, issues


def clear_automaton_cache() -> None:
    """测试与词库热更新后手动失效用。"""
    _AUTOMATON_CACHE.clear()
