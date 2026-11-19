"""Stage 1 / Stage 2 的提示词构造。

集中在一处是为了让「禁止捏造」「保留掩码」「锁定术语」这几条硬约束有单一
修改点。但要清楚：**提示词只是第一道防线**，真正的保证来自 term_guard 的
后置代码级校验。模型遵不遵守提示词是概率问题，门禁拦不拦是确定性的。

掩码占位符 `{{CITE_1}}` 原样出现在提示词里。必须显式告诉模型它们是不可变的
标记而非待填写的模板变量 —— 否则模型会「贴心地」把 {{CITE_1}} 替换成
它猜测的文献名。
"""

from __future__ import annotations

import json

from app.schemas.assist import DraftStrategy, FactTriple, SlotCard
from app.schemas.diagnosis import Lang
from app.services.llm.client import ChatMessage

_MASK_RULE_ZH = (
    "文本中形如 {{CITE_1}}、{{MATH_2}}、{{URL_3}} 的标记是公式与引文的不可变占位符。"
    "必须原样逐字保留，不得改写、翻译、删除、增加或猜测其内容。"
)
_MASK_RULE_EN = (
    "Markers like {{CITE_1}}, {{MATH_2}}, {{URL_3}} are immutable placeholders for "
    "formulas and citations. Reproduce them verbatim; never rewrite, translate, "
    "remove, add, or guess what they contain."
)

_NO_FABRICATION_ZH = (
    "严禁捏造任何事实：数据、指标数值、样本量、数据集名称、年份、文献、机构、"
    "实验条件与案例，凡原文未提供的，一律不得自行编造或「合理推测」。"
)
_NO_FABRICATION_EN = (
    "Never fabricate facts: figures, metric values, sample sizes, dataset names, "
    "years, references, institutions, experimental conditions, or cases. If the "
    "source text does not provide it, do not invent or 'reasonably infer' it."
)


def _mask_rule(lang: Lang) -> str:
    return _MASK_RULE_EN if lang is Lang.EN else _MASK_RULE_ZH


def _no_fabrication(lang: Lang) -> str:
    return _NO_FABRICATION_EN if lang is Lang.EN else _NO_FABRICATION_ZH


def _locked_terms_clause(locked_terms: list[str], lang: Lang) -> str:
    if not locked_terms:
        return ""
    listed = "、".join(locked_terms) if lang is Lang.ZH else ", ".join(locked_terms)
    if lang is Lang.EN:
        return (
            "\nThe following locked technical terms must appear verbatim and at least "
            f"as many times as in the source paragraph: {listed}. "
            "Do not replace them with synonyms, acronyms, or broader categories."
        )
    return (
        f"\n以下锁定术语必须原样出现，且出现次数不少于原段：{listed}。"
        "不得替换为同义词、简称或上位词，也不得用指代词省略。"
    )


# ---------------------------------------------------------------------------
# Stage 1：抽骨（低温提取事实 + 识别空洞表达）
# ---------------------------------------------------------------------------

_DECONSTRUCT_SCHEMA = {
    "facts": [{"subject": "...", "predicate": "...", "obj": "...", "source_quote": "..."}],
    "slots": [
        {
            "kind": "numeric | dataset | metric | citation | case | condition | other",
            "prompt": "向作者提出的具体问题",
            "rationale": "触发该槽位的原文片段",
        }
    ],
}


def build_deconstruct_messages(
    paragraph: str,
    lang: Lang,
    locked_terms: list[str] | None = None,
) -> list[ChatMessage]:
    """Stage 1 提示词：抽事实三元组 + 为空洞表达生成填空槽位。"""
    terms = locked_terms or []
    schema = json.dumps(_DECONSTRUCT_SCHEMA, ensure_ascii=False, indent=2)

    if lang is Lang.EN:
        system = (
            "You are an academic writing analyst. You extract verifiable facts and "
            "identify vague claims that lack concrete support.\n\n"
            f"{_no_fabrication(Lang.EN)}\n{_mask_rule(Lang.EN)}\n\n"
            "For every vague or unsupported claim, emit a slot asking the author for "
            "the missing specifics instead of filling it in yourself. A slot is the "
            "correct output whenever you would otherwise have to guess."
            f"{_locked_terms_clause(terms, Lang.EN)}\n\n"
            f"Reply with JSON only, matching this shape:\n{schema}"
        )
        user = (
            "Extract the core facts from the paragraph below, and emit a slot for each "
            "vague claim that lacks data, a citation, or a concrete case.\n\n"
            f"Paragraph:\n{paragraph}"
        )
    else:
        system = (
            "你是学术写作分析助手，负责抽取可验证的事实，并指出缺乏具体支撑的空洞表述。\n\n"
            f"{_no_fabrication(Lang.ZH)}\n{_mask_rule(Lang.ZH)}\n\n"
            "对每一处空洞或无源的论断，生成一个向作者提问的填空槽位，"
            "而不要自行补全。只要你需要「猜」，正确的输出就是槽位。"
            f"{_locked_terms_clause(terms, Lang.ZH)}\n\n"
            f"只输出 JSON，格式如下：\n{schema}"
        )
        user = (
            "请抽取下面段落的核心事实，并为每一处缺少数据、引文或具体案例的"
            f"空洞表述生成填空槽位。\n\n段落：\n{paragraph}"
        )

    return [ChatMessage("system", system), ChatMessage("user", user)]


# ---------------------------------------------------------------------------
# Stage 2：重构（三种策略并行生成）
# ---------------------------------------------------------------------------

_STRATEGY_ZH: dict[DraftStrategy, str] = {
    DraftStrategy.PLAIN: (
        "【直叙版】按事实的逻辑顺序平实叙述。不使用显性连接词（此外、因此、"
        "综上所述等），靠主语指代与因果关系本身连句。不写段首铺垫句，"
        "不写段尾意义升华句。"
    ),
    DraftStrategy.VARIED: (
        "【错落版】刻意打破句长与句式的均匀：长短句交替，至少包含一个 10 字"
        "以内的短判断句，且相邻句的主干结构不得雷同。同样不使用显性连接词，"
        "不写帽句与升华句。"
    ),
    DraftStrategy.BLIND_OUTLINE: (
        "【盲写提纲】不要写成段落，只输出分点提纲：每点一行，以「- 」开头，"
        "只保留事实骨架与待补数据的位置，供作者自行扩写。"
    ),
}

_STRATEGY_EN: dict[DraftStrategy, str] = {
    DraftStrategy.PLAIN: (
        "[Plain] State the facts in logical order. Use no explicit connectives "
        "(moreover, therefore, in conclusion). Let subject reference and causality "
        "carry the links. No opening scene-setting and no closing significance claim."
    ),
    DraftStrategy.VARIED: (
        "[Varied] Deliberately break uniform sentence length and structure: alternate "
        "long and short sentences, include at least one clause under eight words, and "
        "ensure no two adjacent sentences share the same main structure. Still no "
        "explicit connectives, no opener, no closing elevation."
    ),
    DraftStrategy.BLIND_OUTLINE: (
        "[Outline] Do not write prose. Emit a bullet outline, one point per line "
        "starting with '- ', keeping only the factual skeleton and the positions where "
        "data is still missing, for the author to expand."
    ),
}


def _slot_clause(slots: list[SlotCard], lang: Lang) -> str:
    """把槽位分为「已填」与「跳过」两类分别交代。

    跳过的槽位最关键：必须明确要求模型保留空缺标注，而不是用形容词把空缺
    糊过去 —— 「显著提升」正是这样产生的。
    """
    filled = [s for s in slots if s.is_filled]
    skipped = [s for s in slots if not s.is_filled]
    parts: list[str] = []

    if filled:
        if lang is Lang.EN:
            lines = "\n".join(f"- {s.prompt}: {s.value}" for s in filled)
            parts.append(f"Author-supplied data (use exactly these values):\n{lines}")
        else:
            lines = "\n".join(f"- {s.prompt}：{s.value}" for s in filled)
            parts.append(f"作者已填写的真实数据（必须原样使用这些数值）：\n{lines}")

    if skipped:
        if lang is Lang.EN:
            lines = "\n".join(f"- {s.prompt}" for s in skipped)
            parts.append(
                "The author skipped the following. Do NOT invent values and do NOT "
                "paper over the gap with vague adjectives. Either omit the claim or "
                "mark the gap as [TO BE SUPPLIED]:\n" + lines
            )
        else:
            lines = "\n".join(f"- {s.prompt}" for s in skipped)
            parts.append(
                "作者跳过了以下槽位。不得编造数值，也不得用「显著」「大幅」这类"
                "空泛形容词把空缺糊过去。要么不写该论断，要么以「[待补充]」标注空缺：\n" + lines
            )

    return "\n\n".join(parts)


def build_restructure_messages(
    strategy: DraftStrategy,
    outline: str,
    facts: list[FactTriple],
    slots: list[SlotCard],
    locked_terms: list[str],
    lang: Lang,
    guard_feedback: str = "",
) -> list[ChatMessage]:
    """Stage 2 提示词。guard_feedback 非空时为门禁未通过后的修正重试。"""
    strategy_rule = (_STRATEGY_EN if lang is Lang.EN else _STRATEGY_ZH)[strategy]

    fact_lines = "\n".join(f"- {f.subject} | {f.predicate} | {f.obj}" for f in facts) or (
        outline or "(无)"
    )
    slot_clause = _slot_clause(slots, lang)

    if lang is Lang.EN:
        system = (
            "You rewrite academic prose so that it reads as human-written: concrete, "
            "uneven in rhythm, free of filler connectives.\n\n"
            f"{_no_fabrication(Lang.EN)}\n{_mask_rule(Lang.EN)}\n\n"
            f"{strategy_rule}"
            f"{_locked_terms_clause(locked_terms, Lang.EN)}\n\n"
            "Output the rewritten text only, with no preamble or explanation."
        )
        user_parts = [f"Facts to convey:\n{fact_lines}"]
        if slot_clause:
            user_parts.append(slot_clause)
        if guard_feedback:
            user_parts.append(
                "Your previous draft was rejected by an automated checker for these "
                f"reasons. Fix every one of them:\n{guard_feedback}"
            )
    else:
        system = (
            "你负责重写学术文本，使其读起来像人写的：具体、节奏不均匀、不靠套话连接。\n\n"
            f"{_no_fabrication(Lang.ZH)}\n{_mask_rule(Lang.ZH)}\n\n"
            f"{strategy_rule}"
            f"{_locked_terms_clause(locked_terms, Lang.ZH)}\n\n"
            "只输出重写后的正文，不要任何前言、解释或自我评价。"
        )
        user_parts = [f"需要表达的事实：\n{fact_lines}"]
        if slot_clause:
            user_parts.append(slot_clause)
        if guard_feedback:
            user_parts.append(
                f"你上一版草稿被自动校验拒绝，原因如下，请逐条修正：\n{guard_feedback}"
            )

    return [ChatMessage("system", system), ChatMessage("user", "\n\n".join(user_parts))]
