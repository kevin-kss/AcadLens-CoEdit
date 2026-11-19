"""术语硬门禁单测。

验收指标要求锁定术语保留率与非正文无损率均为 **100%**，因此这里覆盖的重点是
「该拦的必须拦住」：同义替换、简称、上位词、删除、编造标记。
"""

from __future__ import annotations

import pytest

from app.schemas.terms import TermGuardViolationType
from app.services.guardrails.term_guard import validate_locked_terms_preserved

ORIGINAL = "多头自注意力机制在 {{CITE_1}} 中提出，其复杂度见 {{MATH_1}}，RMSE 降低 2.4%。"
LOCKED = ["多头自注意力机制", "RMSE"]


def types_of(result: object) -> set[TermGuardViolationType]:
    return {v.violation_type for v in result.violations}  # type: ignore[attr-defined]


class TestPassingCases:
    def test_verbatim_preservation_passes(self):
        candidate = "{{CITE_1}} 提出了多头自注意力机制，复杂度见 {{MATH_1}}；RMSE 降 2.4%。"
        result = validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED)
        assert result.passed
        assert result.violations == []
        assert result.feedback == ""

    def test_reordering_is_allowed(self):
        """门禁只管「是否保留」，不管语序 —— 重构本就要改语序。"""
        candidate = "RMSE 降 2.4%，见 {{MATH_1}}；多头自注意力机制由 {{CITE_1}} 提出。"
        assert validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED).passed

    def test_extra_occurrences_allowed(self):
        """保留得更多不算违规，只有少于原段才拦。"""
        candidate = (
            "多头自注意力机制见 {{CITE_1}}；多头自注意力机制的复杂度见 {{MATH_1}}，RMSE 降 2.4%。"
        )
        assert validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED).passed

    def test_empty_locked_list_only_checks_masks(self):
        candidate = "改写后的句子，见 {{CITE_1}} 与 {{MATH_1}}。"
        assert validate_locked_terms_preserved(ORIGINAL, candidate, []).passed

    def test_term_absent_from_original_not_required(self):
        """不能强迫模型凭空插入本段没有的术语，那会制造幻觉。"""
        result = validate_locked_terms_preserved(
            "本段只谈长尾分布。", "本段讨论长尾分布问题。", ["RMSE", "Transformer"]
        )
        assert result.passed

    def test_no_masks_no_terms(self):
        assert validate_locked_terms_preserved("普通文本。", "改写后的文本。", []).passed


class TestLockedTermViolations:
    @pytest.mark.parametrize(
        ("candidate", "desc"),
        [
            ("{{CITE_1}} 提出了多头注意力，见 {{MATH_1}}；RMSE 降 2.4%。", "上位词替换"),
            ("{{CITE_1}} 提出了 MHSA，见 {{MATH_1}}；RMSE 降 2.4%。", "简称替换"),
            ("{{CITE_1}} 提出了自注意力机制，见 {{MATH_1}}；RMSE 降 2.4%。", "近义替换"),
            ("{{CITE_1}} 提出该机制，见 {{MATH_1}}；RMSE 降 2.4%。", "指代省略"),
        ],
    )
    def test_synonym_substitution_is_blocked(self, candidate: str, desc: str):
        result = validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED)
        assert not result.passed, desc
        assert TermGuardViolationType.LOCKED_TERM_MISSING in types_of(result)

    def test_english_acronym_removal_blocked(self):
        candidate = "{{CITE_1}} 提出多头自注意力机制，见 {{MATH_1}}；误差降 2.4%。"
        result = validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED)
        assert not result.passed
        assert any(v.subject == "RMSE" for v in result.violations)

    def test_partial_count_loss_blocked(self):
        """原段出现两次、候选只剩一次，也算丢失。"""
        original = "RMSE 下降，RMSE 稳定。"
        candidate = "RMSE 下降并稳定。"
        result = validate_locked_terms_preserved(original, candidate, ["RMSE"])
        assert not result.passed
        violation = result.violations[0]
        assert (violation.expected_count, violation.actual_count) == (2, 1)

    def test_violation_reports_counts(self):
        candidate = "注意力方法降低了误差。"
        result = validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED)
        term_violations = [
            v
            for v in result.violations
            if v.violation_type is TermGuardViolationType.LOCKED_TERM_MISSING
        ]
        assert {v.subject for v in term_violations} == {"多头自注意力机制", "RMSE"}
        for v in term_violations:
            assert v.expected_count == 1
            assert v.actual_count == 0


class TestMaskTokenViolations:
    def test_missing_citation_blocked(self):
        candidate = "多头自注意力机制的复杂度见 {{MATH_1}}，RMSE 降低 2.4%。"
        result = validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED)
        assert not result.passed
        assert TermGuardViolationType.MASK_TOKEN_MISSING in types_of(result)
        assert any(v.subject == "{{CITE_1}}" for v in result.violations)

    def test_missing_math_blocked(self):
        candidate = "多头自注意力机制见 {{CITE_1}}，RMSE 降低 2.4%。"
        result = validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED)
        assert not result.passed
        assert any(v.subject == "{{MATH_1}}" for v in result.violations)

    def test_fabricated_token_blocked(self):
        """模型自行编造引文标记 —— 零捏造合规性要求必须拦下。"""
        candidate = "{{CITE_1}}{{CITE_7}} 提出多头自注意力机制，见 {{MATH_1}}，RMSE 降 2.4%。"
        result = validate_locked_terms_preserved(ORIGINAL, candidate, LOCKED)
        assert not result.passed
        assert TermGuardViolationType.MASK_TOKEN_ALTERED in types_of(result)
        assert any(v.subject == "{{CITE_7}}" for v in result.violations)

    def test_duplicated_token_blocked(self):
        original = "见 {{CITE_1}}。"
        candidate = "见 {{CITE_1}} 与 {{CITE_1}}。"
        result = validate_locked_terms_preserved(original, candidate, [])
        assert not result.passed
        violation = result.violations[0]
        assert violation.violation_type is TermGuardViolationType.MASK_TOKEN_ALTERED
        assert (violation.expected_count, violation.actual_count) == (1, 2)

    def test_altered_token_id_blocked_both_ways(self):
        """{{CITE_1}} 被改成 {{CITE_2}}：既是遗漏也是新增，两条都应报。"""
        original = "见 {{CITE_1}}。"
        candidate = "见 {{CITE_2}}。"
        result = validate_locked_terms_preserved(original, candidate, [])
        assert not result.passed
        assert types_of(result) == {
            TermGuardViolationType.MASK_TOKEN_MISSING,
            TermGuardViolationType.MASK_TOKEN_ALTERED,
        }

    def test_repeated_token_count_must_match(self):
        original = "见 {{CITE_1}} 与 {{CITE_1}}。"
        candidate = "见 {{CITE_1}}。"
        result = validate_locked_terms_preserved(original, candidate, [])
        assert not result.passed
        assert (result.violations[0].expected_count, result.violations[0].actual_count) == (
            2,
            1,
        )


class TestRegexMetacharacterTerms:
    """锁定术语可能含正则元字符，用 str.count 而非 re 才不会静默失配。"""

    @pytest.mark.parametrize("term", ["f(x)", "C++", "p<0.05", "L2-norm", "a.b", "x[i]"])
    def test_metacharacter_term_preserved(self, term: str):
        original = f"定义 {term} 如上。"
        assert validate_locked_terms_preserved(original, f"我们定义 {term}。", [term]).passed

    @pytest.mark.parametrize("term", ["f(x)", "C++", "p<0.05", "L2-norm", "a.b", "x[i]"])
    def test_metacharacter_term_removal_blocked(self, term: str):
        original = f"定义 {term} 如上。"
        result = validate_locked_terms_preserved(original, "我们给出定义。", [term])
        assert not result.passed
        assert any(v.subject == term for v in result.violations)

    def test_dot_is_not_treated_as_wildcard(self):
        """若用未转义的正则，"a.b" 会错误匹配 "axb" 而误判通过。"""
        result = validate_locked_terms_preserved("定义 a.b 如上。", "我们定义 axb。", ["a.b"])
        assert not result.passed


class TestEdgeCases:
    def test_empty_term_ignored(self):
        assert validate_locked_terms_preserved(ORIGINAL, ORIGINAL, ["", "RMSE"]).passed

    def test_duplicate_locked_terms_reported_once(self):
        original = "RMSE 降低。"
        result = validate_locked_terms_preserved(original, "误差降低。", ["RMSE", "RMSE"])
        assert len([v for v in result.violations if v.subject == "RMSE"]) == 1

    def test_empty_candidate_blocks_everything(self):
        result = validate_locked_terms_preserved(ORIGINAL, "", LOCKED)
        assert not result.passed
        assert len(result.violations) == 4  # 2 mask + 2 term

    def test_empty_original_passes(self):
        assert validate_locked_terms_preserved("", "任意草稿。", LOCKED).passed


class TestFeedbackForRetry:
    """违规项要能直接拼成反馈回灌给模型做修正重试。"""

    def test_feedback_empty_when_passed(self):
        assert validate_locked_terms_preserved(ORIGINAL, ORIGINAL, LOCKED).feedback == ""

    def test_feedback_lists_every_violation(self):
        result = validate_locked_terms_preserved(ORIGINAL, "注意力方法降低了误差。", LOCKED)
        feedback = result.feedback
        assert feedback.count("\n") == len(result.violations) - 1
        for v in result.violations:
            assert v.subject in feedback

    def test_feedback_names_the_terms_and_tokens(self):
        result = validate_locked_terms_preserved(ORIGINAL, "注意力方法降低了误差。", LOCKED)
        for expected in ("{{CITE_1}}", "{{MATH_1}}", "多头自注意力机制", "RMSE"):
            assert expected in result.feedback
