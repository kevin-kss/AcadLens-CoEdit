"""轨道 C 单测：套话扫描、具象锚点、空洞度与帽句规则。"""

from __future__ import annotations

import pytest

from app.schemas.diagnosis import IssueType, Lang
from app.schemas.document import SentenceSpan
from app.services.diagnostics.lexicon_loader import load_lexicon
from app.services.diagnostics.track_c_lexicon import (
    compute_fluff_score,
    count_concrete_anchors,
    diagnose_sentence_track_c,
    diagnose_track_c,
)
from app.services.parser.splitter import split_sentences


def span(text: str, start: int = 0) -> SentenceSpan:
    return SentenceSpan(index=0, text=text, start_char=start, end_char=start + len(text))


class TestConcreteAnchors:
    @pytest.mark.parametrize(
        "text",
        [
            "准确率提升了 4.2%。",
            "样本量 n = 120。",
            "显著性满足 p < 0.05。",
            "该结论见 {{CITE_1}}。",
            "推导过程见 {{MATH_3}}。",
            "训练耗时 320 ms。",
            "见 Table 2 的对比。",
            "数据采集于 2024 年。",
        ],
    )
    def test_anchor_detected(self, text: str):
        assert count_concrete_anchors(text) >= 1

    def test_no_anchor_in_pure_abstraction(self):
        assert count_concrete_anchors("本方法显著提升了模型的综合性能与适用范围。") == 0

    def test_overlapping_anchors_merged(self):
        """4.2% 同时命中百分比与兜底数字规则，只应计一次。"""
        assert count_concrete_anchors("提升 4.2%") == 1

    def test_multiple_distinct_anchors(self):
        assert count_concrete_anchors("在 {{CITE_1}} 的设置下，n = 120，准确率为 93.5%。") >= 3


class TestFluffScore:
    def test_anchors_dilute_fluff(self):
        assert compute_fluff_score(4, 0) == 4.0
        assert compute_fluff_score(4, 1) == pytest.approx(4 / 3)
        assert compute_fluff_score(4, 2) == 0.8

    def test_zero_abstract_is_zero(self):
        assert compute_fluff_score(0, 0) == 0.0


class TestChineseLexicon:
    def test_connector_detected(self):
        lex = load_lexicon(Lang.ZH)
        res = diagnose_sentence_track_c(span("值得注意的是，该方法有效。"), lex)
        assert res.connector_hits >= 1
        assert any(i.issue_type is IssueType.AI_CONNECTOR for i in res.issues)

    def test_longest_match_wins(self):
        """命中「综上所述」后不应再单独报其子串。"""
        lex = load_lexicon(Lang.ZH)
        res = diagnose_sentence_track_c(span("综上所述，模型表现稳定。"), lex)
        connector_issues = [i for i in res.issues if i.issue_type is IssueType.AI_CONNECTOR]
        assert len(connector_issues) == 1
        assert connector_issues[0].matched_text == "综上所述"

    def test_low_info_density_triggered(self):
        lex = load_lexicon(Lang.ZH)
        text = "本研究通过多维度的系统化分析，有效提升了治理效能，具有重要意义。"
        res = diagnose_sentence_track_c(span(text), lex)
        assert res.abstract_hits >= 2
        assert res.concrete_anchors == 0
        assert any(i.issue_type is IssueType.LOW_INFO_DENSITY for i in res.issues)

    def test_anchor_suppresses_low_info(self):
        """同样的抽象词，有了数据支撑就不再判为空洞。"""
        lex = load_lexicon(Lang.ZH)
        text = "本方法有效提升了准确率，在 CIFAR-100 上 Top-1 误差降低 2.4%。"
        res = diagnose_sentence_track_c(span(text), lex)
        assert res.concrete_anchors >= 1
        assert not any(i.issue_type is IssueType.LOW_INFO_DENSITY for i in res.issues)

    def test_head_pattern_only_at_head(self):
        lex = load_lexicon(Lang.ZH)
        text = "随着人工智能技术的不断发展，相关研究日益增多。"
        at_head = diagnose_sentence_track_c(span(text), lex, is_head=True)
        not_head = diagnose_sentence_track_c(span(text), lex, is_head=False)
        head_hits = [i for i in at_head.issues if i.issue_type is IssueType.HAT_BOOT_CLICHE]
        assert head_hits
        assert len(head_hits) > len(
            [i for i in not_head.issues if i.issue_type is IssueType.HAT_BOOT_CLICHE]
        )

    def test_tail_pattern_only_at_tail(self):
        lex = load_lexicon(Lang.ZH)
        text = "该研究具有重要的理论意义。"
        at_tail = diagnose_sentence_track_c(span(text), lex, is_tail=True)
        assert any(i.issue_type is IssueType.HAT_BOOT_CLICHE for i in at_tail.issues)

    def test_clean_sentence_has_no_issues(self):
        lex = load_lexicon(Lang.ZH)
        text = "我们在 ImageNet 上训练 90 个 epoch，批大小为 256。"
        res = diagnose_sentence_track_c(span(text), lex)
        assert res.issues == []


class TestEnglishLexicon:
    def test_word_boundary_prevents_false_positive(self):
        """'thus' 不应命中 'enthusiasm'，'so' 不应命中 'solution'。"""
        lex = load_lexicon(Lang.EN)
        res = diagnose_sentence_track_c(
            span("The enthusiasm around this solution was notable in 2021."), lex
        )
        matched = {i.matched_text.lower() for i in res.issues}
        assert "thus" not in matched
        assert "so" not in matched

    def test_real_connector_detected(self):
        lex = load_lexicon(Lang.EN)
        res = diagnose_sentence_track_c(span("Moreover, the model converged faster."), lex)
        assert any(
            i.issue_type is IssueType.AI_CONNECTOR and i.matched_text.lower() == "moreover"
            for i in res.issues
        )

    def test_llm_vocabulary_flagged(self):
        lex = load_lexicon(Lang.EN)
        res = diagnose_sentence_track_c(
            span("We delve into the pivotal role of attention mechanisms."), lex
        )
        assert res.issues

    def test_case_insensitive(self):
        lex = load_lexicon(Lang.EN)
        res = diagnose_sentence_track_c(span("FURTHERMORE, results hold."), lex)
        assert any(i.issue_type is IssueType.AI_CONNECTOR for i in res.issues)


class TestOffsetCorrectness:
    def test_offsets_map_to_paragraph_coordinates(self):
        """句内命中必须换算为段落坐标 —— 前端下划线定位直接依赖这一点。"""
        text = "模型收敛了。值得注意的是，该方法有效。"
        spans = split_sentences(text, Lang.ZH)
        _, _, issues = diagnose_track_c(spans, Lang.ZH)
        assert issues
        for issue in issues:
            assert text[issue.start_char : issue.end_char] == issue.matched_text

    def test_english_offsets_map_back(self):
        text = "The loss dropped. Moreover, accuracy improved by 3.1%."
        spans = split_sentences(text, Lang.EN)
        _, _, issues = diagnose_track_c(spans, Lang.EN)
        for issue in issues:
            assert text[issue.start_char : issue.end_char] == issue.matched_text


class TestParagraphLevel:
    def test_ai_style_paragraph_scores_higher_than_clean(self):
        ai_text = (
            "在当今社会，人工智能的发展日益重要。"
            "值得注意的是，该技术具有多维度的应用价值。"
            "此外，其显著提升了行业的整体效能。"
            "综上所述，该研究具有重要的理论意义与广阔的应用前景。"
        )
        clean_text = (
            "我们在 ImageNet-1k 上训练 ResNet-50，批大小 256，学习率 0.1。"
            "训练 90 个 epoch 后 Top-1 准确率为 76.3%。"
            "去掉残差连接后该数值降至 71.8%。"
        )
        ai_score, _, ai_issues = diagnose_track_c(split_sentences(ai_text, Lang.ZH), Lang.ZH)
        clean_score, _, clean_issues = diagnose_track_c(
            split_sentences(clean_text, Lang.ZH), Lang.ZH
        )
        assert ai_score > clean_score
        assert len(ai_issues) > len(clean_issues)

    def test_empty_input(self):
        assert diagnose_track_c([], Lang.ZH) == (0.0, [], [])

    def test_score_within_bounds(self):
        text = "值得注意的是，此外，综上所述，总而言之，由此可见，显而易见。"
        score, _, _ = diagnose_track_c(split_sentences(text, Lang.ZH), Lang.ZH)
        assert 0.0 <= score <= 100.0
