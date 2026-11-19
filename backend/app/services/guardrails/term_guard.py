"""锁定术语与掩码 Token 的后置硬校验门禁。

这是生成管线的 Quality Gate：Stage 2 产出的每份候选草稿都必须先过这里，
不通过则把结构化的违规项回灌给模型做一次修正重试，仍不通过就在前端置灰。

两条硬规则（均为确定性字符串校验，不依赖 Prompt 遵从）：
1. 原段中出现过的锁定术语，候选草稿中必须至少保留同样的出现次数 ——
   不允许被同义替换、简称或改写；
2. 原段中的公式与引文掩码 Token 必须 100% 原样保留，数量一致。

验收指标要求锁定术语保留率与非正文无损率均为 100%，所以这里宁可误报
（让模型多重试一次）也不漏报。
"""

from __future__ import annotations

from collections import Counter

from app.schemas.terms import (
    TermGuardResult,
    TermGuardViolation,
    TermGuardViolationType,
)
from app.services.parser.masker import find_mask_tokens


def _count_occurrences(haystack: str, needle: str) -> int:
    """重叠无关的子串计数。

    用 str.count 而非正则：锁定术语可能含正则元字符（如 "f(x)"、"C++"），
    用 re 需要转义，漏转义就会在生产中静默失配。
    """
    if not needle:
        return 0
    return haystack.count(needle)


def validate_locked_terms_preserved(
    original_masked: str,
    candidate_masked: str,
    locked_terms: list[str],
) -> TermGuardResult:
    """确定性硬门禁。

    original_masked / candidate_masked 都必须是**带掩码**的文本：掩码 Token
    的一致性正是要校验的内容之一，传还原后的原文会让该规则完全失效。

    locked_terms 中未在原段出现的术语不作要求 —— 不能强迫模型凭空插入
    本段没有的术语，那会制造幻觉。
    """
    violations: list[TermGuardViolation] = []

    # --- 规则 2：掩码 Token 必须原样保留且数量一致 ---
    orig_tokens = Counter(find_mask_tokens(original_masked))
    cand_tokens = Counter(find_mask_tokens(candidate_masked))
    for token, expected in sorted(orig_tokens.items()):
        actual = cand_tokens.get(token, 0)
        if actual < expected:
            kind = "引文" if token.startswith("{{CITE") else "公式或非正文标记"
            violations.append(
                TermGuardViolation(
                    violation_type=TermGuardViolationType.MASK_TOKEN_MISSING,
                    subject=token,
                    expected_count=expected,
                    actual_count=actual,
                    message=(
                        f"遗漏{kind} {token}：原段出现 {expected} 次，"
                        f"候选草稿仅 {actual} 次。该占位符必须原样保留，不得改写或删除。"
                    ),
                )
            )

    # 候选中出现原段没有的 Token：模型自行编造了引文或公式标记
    for token, actual in sorted(cand_tokens.items()):
        expected = orig_tokens.get(token, 0)
        if actual > expected:
            violations.append(
                TermGuardViolation(
                    violation_type=TermGuardViolationType.MASK_TOKEN_ALTERED,
                    subject=token,
                    expected_count=expected,
                    actual_count=actual,
                    message=(
                        f"擅自新增非正文标记 {token}：原段出现 {expected} 次，"
                        f"候选草稿为 {actual} 次。不得编造引文或公式占位符。"
                    ),
                )
            )

    # --- 规则 1：原段出现过的锁定术语必须保留 ---
    seen: set[str] = set()
    for term in locked_terms:
        if not term or term in seen:
            continue
        seen.add(term)
        expected = _count_occurrences(original_masked, term)
        if expected == 0:
            continue  # 原段没有，不要求候选插入
        actual = _count_occurrences(candidate_masked, term)
        if actual < expected:
            violations.append(
                TermGuardViolation(
                    violation_type=TermGuardViolationType.LOCKED_TERM_MISSING,
                    subject=term,
                    expected_count=expected,
                    actual_count=actual,
                    message=(
                        f"核心术语「{term}」被改动或丢失：原段出现 {expected} 次，"
                        f"候选草稿仅 {actual} 次。该术语已锁定，必须原样保留，"
                        "不得替换为同义词、简称或上位词。"
                    ),
                )
            )

    return TermGuardResult(passed=not violations, violations=violations)
