"""中英文学术分句器。

与通用分句器的区别在于**否决规则**：学术文本里的句号大量不是句子边界 ——
`Fig. 1`、`Smith et al.`、`e.g.`、`3.14`、`Sec. 4.2`、`Ph.D.`、`J. Smith`。
误切一处，该句的句长会被腰斩，直接污染 Track B 的句长变异系数与依存骨架。

实现为单遍扫描 + 否决表，不依赖模型，因此可在 60ms 实时重算预算内运行。
所有返回的 SentenceSpan 都带源文本精确偏移，供上游把句内命中坐标换算回段落坐标。
"""

from __future__ import annotations

import re

from app.schemas.diagnosis import Lang
from app.schemas.document import SentenceSpan

# ---------------------------------------------------------------------------
# 终止符与收尾符
# ---------------------------------------------------------------------------
_CJK_ENDERS = frozenset("。！？；…")
_LATIN_ENDERS = frozenset(".!?")
_ALL_ENDERS = _CJK_ENDERS | _LATIN_ENDERS

# 句末可跟随的收尾符：引号、各类右括号
_TRAILING_CLOSERS = frozenset("”’\"'）)」』】〕》>")

_MASK_TOKEN_RE = re.compile(r"\{\{[A-Z_]+_\d+\}\}")

# ---------------------------------------------------------------------------
# 缩写否决表
# ---------------------------------------------------------------------------
# 硬否决：这些缩写后面几乎必然接同句内容（`Fig. 1`、`e.g. the`、`Dr. Smith`），
# 无论后续是否大写都不切。
_HARD_ABBREV_RE = re.compile(
    r"(?:^|[\s(\[\"'“‘])(?:"
    r"Fig|Figs|Figure|Tab|Tabs|Table|Eq|Eqs|Eqn|Ref|Refs|Sec|Secs|Sect|Chap|Ch"
    r"|No|Nos|Vol|Vols|pp|p|Para|App|Alg|Thm|Def|Lem|Cor|Prop|Rem"
    r"|Dr|Prof|Mr|Mrs|Ms|Messrs|St|Jr|Sr|Rev|Hon"
    r"|vs|cf|viz|approx|resp|ca|incl|excl|min|max|avg|std|var|Est"
    r"|Inc|Ltd|Co|Corp|Univ|Dept|Lab|Eds|ed|edn|Trans|Repr"
    r")\.$",
    re.IGNORECASE,
)

# 点分缩写：e.g. / i.e. / U.S. / Ph.D. / a.m. —— 末尾那个点同样不是边界
_DOTTED_ABBREV_RE = re.compile(
    r"(?:^|[\s(\[\"'“‘])(?:"
    r"[A-Za-z]\.(?:[A-Za-z]\.)+"  # e.g.  i.e.  U.S.  a.m.
    r"|Ph\.D|M\.Sc|B\.Sc|M\.A|B\.A|M\.D"
    r")$",
    re.IGNORECASE,
)

# 人名首字母缩写：`J. Smith`、`A. B. Chen`（区分大小写，避免误伤句末单字母）
_INITIAL_RE = re.compile(r"(?:^|[\s(\[\"'“‘])[A-Z]\.$")

# 软否决：`etc.` / `et al.` 既可能句中也可能句末 ——
# 仅当后续是明确的新句开头（大写字母或中文）时才切。
_SOFT_ABBREV_RE = re.compile(
    r"(?:^|[\s(\[\"'“‘])(?:etc|et\s+al|et\s+seq|ibid|op\.\s*cit)\.$",
    re.IGNORECASE,
)

# 可作为新句首字符：大写拉丁字母、中文、数字、左引号/左括号、掩码起始
_SENT_START_RE = re.compile(r"[A-Z一-鿿㐀-䶿“‘\"'(\[（【「『]|\{\{")


def _mask_ranges(text: str) -> list[tuple[int, int]]:
    """掩码 Token 的 [start, end) 区间。Token 内部任何字符都不得作为切分点。"""
    return [(m.start(), m.end()) for m in _MASK_TOKEN_RE.finditer(text)]


def _in_ranges(pos: int, ranges: list[tuple[int, int]]) -> tuple[bool, int]:
    for start, end in ranges:
        if start <= pos < end:
            return True, end
    return False, pos


def _next_visible_index(text: str, pos: int) -> int:
    """跳过空白后的首个非空白字符下标；全为空白则返回 len(text)。"""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _next_visible(text: str, pos: int) -> str:
    """跳过空白后的首个字符；到末尾返回空串。"""
    idx = _next_visible_index(text, pos)
    return text[idx] if idx < len(text) else ""


def _is_boundary(text: str, ender_start: int, ender_end: int, after: int) -> bool:
    """判定一处终止符是否为真实句子边界。

    ender_start / ender_end：终止符连续串的 [start, end)。
    after：收尾符（引号、右括号）之后的位置，即候选句尾。
    """
    run = text[ender_start:ender_end]

    # 中文终止符语义明确，直接成立
    if any(c in _CJK_ENDERS for c in run):
        return True

    nxt = _next_visible(text, after)
    head = text[:ender_end]

    # `!` `?`：后接小写拉丁字母说明仍在句中（如 "Why? because..." 式行文少见但保守处理）
    if "!" in run or "?" in run:
        return nxt == "" or not nxt.islower()

    # 以下均为单纯的 `.`（含 `...`）
    prev = text[ender_start - 1] if ender_start > 0 else ""

    # 小数与版本号：3.14、v1.2、Sec 4.2
    if prev.isdigit() and nxt.isdigit():
        return False

    # 点后紧跟非空白（www.example.com、filename.txt）不是边界
    if after < len(text) and not text[after].isspace():
        return False

    if nxt == "":
        return True

    if _HARD_ABBREV_RE.search(head) or _DOTTED_ABBREV_RE.search(head):
        return False

    if _INITIAL_RE.search(head):
        return False

    if _SOFT_ABBREV_RE.search(head):
        # `et al. (2024)` / `etc. {{CITE_1}}` 仍在句中；`et al. The results` 是新句
        return bool(nxt.isupper() or re.match(r"[一-鿿]", nxt))

    return bool(_SENT_START_RE.match(text, _next_visible_index(text, after)))


def split_sentences(text: str, lang: Lang | str = Lang.ZH) -> list[SentenceSpan]:
    """把段落切为带偏移的句子列表。

    lang 当前不改变切分规则（规则对中英混排同时生效），保留参数是为了后续
    按语种微调否决表时不动调用方签名。
    """
    _ = lang
    spans: list[SentenceSpan] = []
    masks = _mask_ranges(text)
    length = len(text)

    cursor = 0  # 当前句起点（含）
    pos = 0
    while pos < length:
        in_mask, mask_end = _in_ranges(pos, masks)
        if in_mask:
            pos = mask_end
            continue

        if text[pos] not in _ALL_ENDERS:
            pos += 1
            continue

        # 吞掉连续终止符：`...`、`？！`、`……`
        ender_end = pos + 1
        while ender_end < length and text[ender_end] in _ALL_ENDERS:
            inside, _ = _in_ranges(ender_end, masks)
            if inside:
                break
            ender_end += 1

        # 吞掉收尾引号与右括号
        after = ender_end
        while after < length and text[after] in _TRAILING_CLOSERS:
            after += 1

        if _is_boundary(text, pos, ender_end, after):
            raw = text[cursor:after]
            stripped = raw.strip()
            if stripped:
                offset = cursor + (len(raw) - len(raw.lstrip()))
                spans.append(
                    SentenceSpan(
                        index=len(spans),
                        text=stripped,
                        start_char=offset,
                        end_char=offset + len(stripped),
                    )
                )
            cursor = after
        pos = after

    # 收尾：无终止符的残余文本（摘要末行、列表项等）
    tail = text[cursor:]
    if tail.strip():
        offset = cursor + (len(tail) - len(tail.lstrip()))
        stripped = tail.strip()
        spans.append(
            SentenceSpan(
                index=len(spans),
                text=stripped,
                start_char=offset,
                end_char=offset + len(stripped),
            )
        )

    return spans


def split_paragraphs(raw_text: str) -> list[str]:
    """按空行与硬换行切段，丢弃空白段。"""
    return [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", raw_text) if p.strip()]
