"""聚合引擎单测：权重、段落/全篇诊断与白盒可追溯性。"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.schemas.diagnosis import EngineMode, IssueType, Lang, RiskLevel
from app.schemas.document import ParsedParagraph
from app.services.diagnostics.engine import (
    _aggregate,
    clean_char_count,
    diagnose_document,
    diagnose_paragraph,
)
from app.services.parser.masker import mask_academic_text
from app.services.parser.splitter import split_sentences

AI_STYLE = (
    "在当今社会，人工智能技术的发展日益重要。"
    "值得注意的是，该方法具有多维度的应用价值。"
    "此外，其显著提升了系统的整体效能。"
    "综上所述，本研究具有重要的理论意义与广阔的应用前景。"
)

HUMAN_STYLE = (
    "我们在 ImageNet-1k 上训练 ResNet-50。"
    "批大小 256，学习率 0.1，共 90 个 epoch。"
    "Top-1 准确率为 76.3%。"
    "去掉残差连接后，该数值降至 71.8%，且训练在第 40 个 epoch 后不再收敛。"
)


class TestRiskLevel:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.0, RiskLevel.LOW),
            (34.9, RiskLevel.LOW),
            (35.0, RiskLevel.MEDIUM),
            (65.0, RiskLevel.MEDIUM),
            (65.1, RiskLevel.HIGH),
            (100.0, RiskLevel.HIGH),
        ],
    )
    def test_thresholds_match_spec(self, score: float, expected: RiskLevel):
        assert RiskLevel.from_score(score) is expected


class TestAggregate:
    def test_dual_track_weights(self):
        settings = get_settings()
        risk, mode = _aggregate(80.0, 40.0, None, EngineMode.DUAL_TRACK)
        expected = settings.weight_dual_syntactic * 80 + settings.weight_dual_lexicon * 40
        assert risk == pytest.approx(expected)
        assert mode is EngineMode.DUAL_TRACK

    def test_tri_track_weights(self):
        settings = get_settings()
        risk, mode = _aggregate(80.0, 40.0, 60.0, EngineMode.TRI_TRACK)
        expected = (
            settings.weight_tri_curvature * 60
            + settings.weight_tri_syntactic * 80
            + settings.weight_tri_lexicon * 40
        )
        assert risk == pytest.approx(expected)
        assert mode is EngineMode.TRI_TRACK

    def test_tri_track_without_curvature_falls_back(self):
        """配置声明三轨但 Track A 未产出曲率时，必须退回双轨权重而非静默算错。"""
        risk, mode = _aggregate(80.0, 40.0, None, EngineMode.TRI_TRACK)
        dual_risk, _ = _aggregate(80.0, 40.0, None, EngineMode.DUAL_TRACK)
        assert mode is EngineMode.DUAL_TRACK
        assert risk == pytest.approx(dual_risk)

    def test_score_clamped(self):
        risk, _ = _aggregate(100.0, 100.0, 100.0, EngineMode.TRI_TRACK)
        assert 0.0 <= risk <= 100.0


class TestDiagnoseParagraph:
    def test_ai_style_scores_higher_than_human_style(self):
        ai = diagnose_paragraph("p1", AI_STYLE, Lang.ZH)
        human = diagnose_paragraph("p2", HUMAN_STYLE, Lang.ZH)
        assert ai.overall_risk > human.overall_risk

    def test_empty_paragraph_is_safe(self):
        diag = diagnose_paragraph("p0", "", Lang.ZH)
        assert diag.overall_risk == 0.0
        assert diag.risk_level is RiskLevel.LOW
        assert diag.sentences == []
        assert diag.span_issues == []

    def test_whitespace_only_paragraph(self):
        diag = diagnose_paragraph("p0", "   \n  ", Lang.ZH)
        assert diag.sentences == []

    def test_sentence_metrics_align_with_sentences(self):
        diag = diagnose_paragraph("p1", AI_STYLE, Lang.ZH)
        spans = split_sentences(AI_STYLE, Lang.ZH)
        assert len(diag.sentences) == len(spans)
        for metric, span in zip(diag.sentences, spans, strict=True):
            assert metric.text == span.text
            assert metric.start_char == span.start_char
            assert 0 <= metric.sentence_risk <= 100

    def test_track_scores_are_exposed(self):
        """白盒要求：分数必须能回溯到具体轨道。"""
        diag = diagnose_paragraph("p1", AI_STYLE, Lang.ZH)
        assert 0 <= diag.track_scores.syntactic <= 100
        assert 0 <= diag.track_scores.lexicon <= 100
        assert diag.track_scores.curvature is None  # Track A 待 M4 接入

    def test_skeleton_source_recorded(self):
        diag = diagnose_paragraph("p1", AI_STYLE, Lang.ZH)
        assert {m.skeleton_source for m in diag.sentences} <= {"dep", "pos"}

    def test_issues_sorted_by_offset(self):
        diag = diagnose_paragraph("p1", AI_STYLE, Lang.ZH)
        offsets = [(i.start_char, i.end_char) for i in diag.span_issues]
        assert offsets == sorted(offsets)

    def test_all_issue_offsets_within_text(self):
        diag = diagnose_paragraph("p1", AI_STYLE, Lang.ZH)
        for issue in diag.span_issues:
            assert 0 <= issue.start_char < issue.end_char <= len(AI_STYLE)
            assert AI_STYLE[issue.start_char : issue.end_char] == issue.matched_text

    def test_masked_text_is_diagnosed_without_crash(self):
        raw = "该结论与前人一致 [12]，推导见 $E = mc^2$，准确率提升 4.2%。此外，方法有效。"
        masked = mask_academic_text(raw)
        diag = diagnose_paragraph("p1", masked.masked, Lang.ZH)
        assert diag.sentences
        for issue in diag.span_issues:
            assert masked.masked[issue.start_char : issue.end_char] == issue.matched_text

    def test_english_paragraph(self):
        text = (
            "In today's rapidly evolving landscape, attention is pivotal. "
            "Moreover, the model delves into multifaceted representations. "
            "Furthermore, it significantly improves performance. "
            "In conclusion, this work provides valuable insights."
        )
        diag = diagnose_paragraph("p-en", text, Lang.EN)
        assert diag.lang is Lang.EN
        assert diag.overall_risk > 20
        assert any(i.issue_type is IssueType.AI_CONNECTOR for i in diag.span_issues)

    def test_reused_sentences_give_same_result(self):
        spans = split_sentences(AI_STYLE, Lang.ZH)
        a = diagnose_paragraph("p1", AI_STYLE, Lang.ZH)
        b = diagnose_paragraph("p1", AI_STYLE, Lang.ZH, sentences=spans)
        assert a.overall_risk == pytest.approx(b.overall_risk)

    def test_explicit_mode_overrides_settings(self):
        diag = diagnose_paragraph("p1", AI_STYLE, Lang.ZH, mode=EngineMode.DUAL_TRACK)
        assert diag.mode is EngineMode.DUAL_TRACK


class TestDiagnoseDocument:
    def _paragraphs(self) -> list[ParsedParagraph]:
        return [
            ParsedParagraph(
                paragraph_id="p1",
                order_index=0,
                section_type="body",
                masked_text=AI_STYLE,
                sentences=split_sentences(AI_STYLE, Lang.ZH),
                lang=Lang.ZH,
            ),
            ParsedParagraph(
                paragraph_id="p2",
                order_index=1,
                section_type="body",
                masked_text=HUMAN_STYLE,
                sentences=split_sentences(HUMAN_STYLE, Lang.ZH),
                lang=Lang.ZH,
            ),
            ParsedParagraph(
                paragraph_id="p3",
                order_index=2,
                section_type="reference_skip",
                masked_text="[1] Smith, J. Attention Is All You Need. 2017.",
                sentences=[],
                lang=Lang.ZH,
            ),
        ]

    def test_reference_paragraphs_skipped(self):
        """参考文献的句长与用词分布与正文完全不同，纳入统计会污染全篇分数。"""
        results, summary = diagnose_document(self._paragraphs())
        assert [r.paragraph_id for r in results] == ["p1", "p2"]
        assert summary.paragraph_count == 2

    def test_summary_aggregates(self):
        results, summary = diagnose_document(self._paragraphs())
        assert summary.sentence_count == sum(len(r.sentences) for r in results)
        assert summary.mean_risk == pytest.approx(
            sum(r.overall_risk for r in results) / len(results)
        )

    def test_issue_counts_tally(self):
        results, summary = diagnose_document(self._paragraphs())
        total = sum(len(r.span_issues) for r in results)
        assert sum(summary.issue_counts.values()) == total

    def test_high_risk_ids_consistent(self):
        results, summary = diagnose_document(self._paragraphs())
        expected = [r.paragraph_id for r in results if r.risk_level is RiskLevel.HIGH]
        assert summary.high_risk_paragraph_ids == expected

    def test_empty_document(self):
        results, summary = diagnose_document([])
        assert results == []
        assert summary.paragraph_count == 0
        assert summary.mean_risk == 0.0


class TestCleanCharCount:
    def test_mask_tokens_excluded(self):
        """掩码剔除后首尾空白一并去掉，故 "摘要 {{CITE_1}}" 计 2 字。"""
        assert clean_char_count("摘要 {{CITE_1}}") == 2

    def test_inner_whitespace_still_counted(self):
        assert clean_char_count("摘要 {{CITE_1}} 结论") == len("摘要  结论")

    def test_plain_text(self):
        assert clean_char_count("  模型收敛  ") == 4
