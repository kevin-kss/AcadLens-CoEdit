"""需要 spaCy 依存模型的 Track B 回归测试。

这些用例守的是两个**静默失效**的 bug —— 它们不抛异常、不改变接口，只是让
排比检测永久失灵，而降级标记仍自称 "dep"：

1. `t.head is t` 判定 ROOT：spaCy 的 Token 是每次访问重建的代理对象，
   `is` 恒为 False，导致所有骨架变成空串 -> detect_syntactic_parallelism 永不触发。
2. 禁用 `attribute_ruler`：它负责 tag_ -> pos_ 映射，禁用后 pos_ 恒为空串。

未安装模型时整类跳过（CI 可只跑降级路径）：uv sync --extra models 后生效。
"""

from __future__ import annotations

import pytest

from app.schemas.diagnosis import IssueType, Lang
from app.services.diagnostics.nlp_provider import get_pipeline
from app.services.diagnostics.track_b_syntactic import (
    build_dep_skeletons,
    detect_syntactic_parallelism,
    diagnose_track_b,
)
from app.services.parser.splitter import split_sentences

pytestmark = pytest.mark.needs_spacy_model


def _require(lang: Lang) -> None:
    if not get_pipeline(lang).available:
        pytest.skip(f"{lang.value} spaCy 模型未安装，跳过真依存测试")


class TestSkeletonsAreNonEmpty:
    @pytest.mark.parametrize(
        ("lang", "sentence"),
        [
            (Lang.ZH, "该方法具有多维度的应用价值。"),
            (Lang.ZH, "我们在 ImageNet-1k 上训练 ResNet-50，批大小为 256。"),
            (Lang.EN, "The model delves into multifaceted representations."),
            (Lang.EN, "We train ResNet-50 on ImageNet with a batch size of 256."),
        ],
    )
    def test_skeleton_not_empty(self, lang: Lang, sentence: str):
        """骨架为空串意味着 ROOT 没找到 —— 排比检测会永久失灵。"""
        _require(lang)
        skeletons, source = build_dep_skeletons([sentence], lang)
        assert source == "dep"
        assert skeletons[0] != "", "依存骨架为空：ROOT 判定失败（勿用 `is` 比较 Token）"

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_skeleton_starts_with_pos_tag(self, lang: Lang):
        """首段必须是 ROOT 的通用词性，而非空串 —— attribute_ruler 不可禁用。"""
        _require(lang)
        sentence = (
            "该方法具有多维度的应用价值。"
            if lang is Lang.ZH
            else "The method yields consistent gains."
        )
        skeleton = build_dep_skeletons([sentence], lang)[0][0]
        head = skeleton.split("-")[0]
        assert head, f"骨架首段为空，pos_ 丢失：{skeleton!r}"
        assert head.isupper(), f"骨架首段应为 UPOS 词性标签，实际为 {head!r}"

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_skeleton_contains_dependency_labels(self, lang: Lang):
        _require(lang)
        sentence = (
            "我们在三个数据集上评估了该模型的泛化能力。"
            if lang is Lang.ZH
            else "We evaluate the generalization of the model on three datasets."
        )
        skeleton = build_dep_skeletons([sentence], lang)[0][0]
        assert "-" in skeleton, f"骨架缺少依存标签序列：{skeleton!r}"


class TestParallelismWithRealParser:
    def test_structurally_identical_zh_sentences_flagged(self):
        """真依存下，同模子的两句必须判为同构。"""
        _require(Lang.ZH)
        sents = [
            "该方法具有多维度的应用价值。",
            "该系统具有广泛的实用价值。",
        ]
        skeletons, source = build_dep_skeletons(sents, Lang.ZH)
        assert source == "dep"
        assert detect_syntactic_parallelism(skeletons) == [0, 1]

    def test_structurally_different_zh_sentences_not_flagged(self):
        _require(Lang.ZH)
        sents = [
            "不对。",
            "我们在 ImageNet-1k 上训练 ResNet-50，批大小为 256，学习率设为 0.1。",
        ]
        skeletons, _ = build_dep_skeletons(sents, Lang.ZH)
        assert detect_syntactic_parallelism(skeletons) == []

    def test_parallel_issue_surfaces_in_diagnosis(self):
        """端到端：排比问题必须出现在 span_issues 中并带正确坐标。"""
        _require(Lang.ZH)
        text = "该方法具有多维度的应用价值。该系统具有广泛的实用价值。该框架具有良好的扩展价值。"
        spans = split_sentences(text, Lang.ZH)
        _, _, skeletons, source, issues = diagnose_track_b(spans, Lang.ZH)
        assert source == "dep"
        assert all(s for s in skeletons)
        parallel = [i for i in issues if i.issue_type is IssueType.SYNTACTIC_PARALLEL]
        assert parallel, "连续同构句未被标记为排比"
        for issue in parallel:
            assert text[issue.start_char : issue.end_char] == issue.matched_text
            assert "来源 dep" in issue.message

    def test_message_reports_real_skeleton_not_na(self):
        """白盒可追溯：提示里不应出现 N/A 占位。"""
        _require(Lang.ZH)
        text = "该方法具有多维度的应用价值。该系统具有广泛的实用价值。"
        spans = split_sentences(text, Lang.ZH)
        _, _, _, _, issues = diagnose_track_b(spans, Lang.ZH)
        for issue in issues:
            if issue.issue_type is IssueType.SYNTACTIC_PARALLEL:
                assert "骨架 N/A" not in issue.message


class TestPipelineConfig:
    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_parser_and_attribute_ruler_active(self, lang: Lang):
        _require(lang)
        handle = get_pipeline(lang)
        assert handle.nlp is not None
        assert "parser" in handle.nlp.pipe_names
        assert "attribute_ruler" in handle.nlp.pipe_names, (
            "attribute_ruler 被禁用会让 Token.pos_ 恒为空串"
        )

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_pipeline_is_cached(self, lang: Lang):
        _require(lang)
        assert get_pipeline(lang) is get_pipeline(lang)
