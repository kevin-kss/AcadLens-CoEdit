"""轨道 B 单测：句长突发性与句法骨架同构度。

骨架来源在无 spaCy 模型的环境下为 "pos"（降级近似），装了模型则为 "dep"。
断言刻意只依赖「同构的句子应被判为同构」这一相对性质，不绑定具体骨架字面值，
以便两种来源下都成立。
"""

from __future__ import annotations

import pytest

from app.schemas.diagnosis import IssueType, Lang
from app.services.diagnostics.track_b_syntactic import (
    build_dep_skeletons,
    compute_burstiness_cv,
    detect_syntactic_parallelism,
    diagnose_track_b,
    effective_length,
)
from app.services.parser.splitter import split_sentences


class TestEffectiveLength:
    def test_chinese_counts_characters_without_punctuation(self):
        assert effective_length("模型收敛了。", Lang.ZH) == 5

    def test_english_counts_words(self):
        assert effective_length("The model converged quickly.", Lang.EN) == 4

    def test_mask_tokens_excluded(self):
        with_mask = effective_length("该结论见 {{CITE_1}}。", Lang.ZH)
        without = effective_length("该结论见 。", Lang.ZH)
        assert with_mask == without == 4

    def test_english_mask_token_excluded(self):
        assert effective_length("As shown in {{CITE_1}} results hold.", Lang.EN) == 5

    def test_empty_string(self):
        assert effective_length("", Lang.ZH) == 0


class TestBurstiness:
    def test_short_paragraph_returns_neutral(self):
        """少于 3 句不做低波动惩罚 —— 两句的方差没有统计意义。"""
        assert compute_burstiness_cv(["很短。", "也短。"], Lang.ZH) == 0.5

    def test_uniform_lengths_give_low_cv(self):
        uniform = ["这是一个十二字的句子啊。"] * 5
        assert compute_burstiness_cv(uniform, Lang.ZH) < 0.05

    def test_varied_lengths_give_high_cv(self):
        varied = [
            "不对。",
            "这个方法在三个数据集上都失效了，原因是梯度在深层网络中迅速衰减。",
            "我们改用残差连接。",
            "随后准确率从百分之六十一上升到百分之八十九，并且在长尾子集上保持稳定。",
        ]
        assert compute_burstiness_cv(varied, Lang.ZH) > 0.3

    def test_all_empty_sentences_safe(self):
        assert compute_burstiness_cv(["", "", ""], Lang.ZH) == 0.0


class TestSkeletonsAndParallelism:
    def test_skeleton_source_is_reported(self):
        skeletons, source = build_dep_skeletons(["模型收敛了。"], Lang.ZH)
        assert source in ("dep", "pos")
        assert len(skeletons) == 1

    def test_empty_input(self):
        assert build_dep_skeletons([], Lang.ZH) == ([], "none")

    def test_identical_skeletons_flagged(self):
        skel = ["ADV-SBV-PREP-VOB-COO"] * 3
        assert detect_syntactic_parallelism(skel) == [0, 1, 2]

    def test_different_skeletons_not_flagged(self):
        skel = ["ADV-SBV-VOB", "ROOT-ATT-DE-QUN-MMM-XXX-YYY"]
        assert detect_syntactic_parallelism(skel) == []

    def test_blank_skeletons_skipped(self):
        assert detect_syntactic_parallelism(["", "", ""]) == []

    def test_very_short_skeletons_skipped(self):
        """单 token 骨架天然同构，不应触发排比告警。"""
        assert detect_syntactic_parallelism(["VERB", "VERB"]) == []

    def test_only_adjacent_pairs_compared(self):
        """首尾同构但中间不同：不算排比，排比的观感来自连续出现。"""
        skel = ["A-B-C-D-E", "Z-Y-X-W-V-U-T", "A-B-C-D-E"]
        assert detect_syntactic_parallelism(skel) == []

    def test_threshold_override(self):
        skel = ["A-B-C-D-E", "A-B-C-D-F"]
        assert detect_syntactic_parallelism(skel, threshold=0.99) == []
        assert detect_syntactic_parallelism(skel, threshold=0.5) == [0, 1]


class TestDiagnoseTrackB:
    def test_uniform_paragraph_flags_uniform_length(self):
        text = (
            "这是第一个十四字的句子。这是第二个十四字句子。"
            "这是第三个十四字句子。这是第四个十四字句子。"
        )
        spans = split_sentences(text, Lang.ZH)
        score, cv, _, _, issues = diagnose_track_b(spans, Lang.ZH)
        assert cv < 0.18
        assert any(i.issue_type is IssueType.UNIFORM_LENGTH for i in issues)
        assert score > 40

    def test_varied_paragraph_scores_low(self):
        text = (
            "不对。"
            "这个方法在三个数据集上都失效了，原因是梯度在深层网络里迅速衰减到接近零。"
            "我们改用残差连接。"
        )
        spans = split_sentences(text, Lang.ZH)
        score, cv, _, _, issues = diagnose_track_b(spans, Lang.ZH)
        assert cv > 0.3
        assert not any(i.issue_type is IssueType.UNIFORM_LENGTH for i in issues)
        assert score < 60

    def test_issue_offsets_within_paragraph(self):
        text = "这是第一个十四字的句子。这是第二个十四字句子。这是第三个十四字句子。"
        spans = split_sentences(text, Lang.ZH)
        _, _, _, _, issues = diagnose_track_b(spans, Lang.ZH)
        for issue in issues:
            assert 0 <= issue.start_char < issue.end_char <= len(text)
            assert text[issue.start_char : issue.end_char] == issue.matched_text

    def test_empty_sentences(self):
        score, cv, skeletons, source, issues = diagnose_track_b([], Lang.ZH)
        assert (score, skeletons, source, issues) == (0.0, [], "none", [])
        assert cv == 0.5

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_single_sentence_no_crash(self, lang: Lang):
        text = "Only one sentence here." if lang is Lang.EN else "只有一个句子。"
        spans = split_sentences(text, lang)
        score, _, _, _, _ = diagnose_track_b(spans, lang)
        assert 0 <= score <= 100
