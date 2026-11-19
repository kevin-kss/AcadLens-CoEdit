"""C-Value 术语预提取单测。

断言尽量表达**相对性质**（真术语应排在碎片之前、嵌套短语应被折减），
而非绑定具体分数 —— 阈值与权重后续要用真实语料重标，绑死数值会让这批
用例在重标时集体转红却说明不了任何问题。

需要 spaCy 模型的用例走 needs_spacy_model 标记；降级路径的用例无条件运行。
"""

from __future__ import annotations

import pytest

from app.schemas.diagnosis import Lang
from app.services.diagnostics.nlp_provider import get_pipeline
from app.services.guardrails.term_extractor import (
    _is_viable,
    _word_count,
    extract_terms,
)

ZH_TEXT = (
    "本文提出一种基于多头自注意力机制的长尾识别方法。多头自注意力机制在 ImageNet-1k 上"
    "表现稳定，但在长尾分布数据集上退化明显。我们改进了多头自注意力机制的权重初始化策略，"
    "并用 RMSE 与 Top-1 准确率评估长尾识别效果。实验表明，改进后的长尾识别方法在 "
    "CIFAR-100 长尾子集上的 RMSE 降低 2.4%。多头自注意力机制的计算开销与长尾分布相关。"
)

EN_TEXT = (
    "This paper proposes a long-tailed recognition method based on multi-head "
    "self-attention. Multi-head self-attention performs stably on ImageNet-1k but "
    "degrades on long-tailed distributions. We revise the weight initialization of "
    "multi-head self-attention and evaluate long-tailed recognition with RMSE and "
    "Top-1 accuracy. Experiments show the revised long-tailed recognition method "
    "lowers RMSE by 2.4% on the CIFAR-100 subset."
)


def terms_of(text: str, lang: Lang) -> list[str]:
    return [c.term for c in extract_terms(text, lang).candidates]


class TestViability:
    @pytest.mark.parametrize(
        "phrase",
        ["2.4", "100", "%", "3.14", ",", "0.05"],
    )
    def test_pure_numbers_rejected(self, phrase: str):
        assert not _is_viable(phrase, Lang.ZH)

    @pytest.mark.parametrize("phrase", ["的", "中", "x"])
    def test_single_character_rejected(self, phrase: str):
        assert not _is_viable(phrase, Lang.ZH)

    def test_newline_spanning_rejected(self):
        """跨行切片会与同名单行短语各自分走频次，使真术语得分被腰斩。"""
        assert not _is_viable("多头自注意力\n机制", Lang.ZH)
        assert not _is_viable("multi-head\nself-attention", Lang.EN)

    @pytest.mark.parametrize("phrase", ["-head self", "multi-", "_norm", "自注意力-"])
    def test_leading_or_trailing_connector_rejected(self, phrase: str):
        assert not _is_viable(phrase, Lang.ZH)

    @pytest.mark.parametrize(
        "phrase", ["多头自注意力机制", "RMSE", "multi-head self-attention", "ImageNet-1k"]
    )
    def test_real_terms_accepted(self, phrase: str):
        assert _is_viable(phrase, Lang.ZH)


class TestWordCount:
    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("多头自注意力机制", 4),
            ("长尾识别", 2),
            ("RMSE", 1),
            ("multi-head self-attention", 2),
        ],
    )
    def test_counts(self, phrase: str, expected: int):
        lang = Lang.EN if phrase[0].isascii() and " " in phrase else Lang.ZH
        assert _word_count(phrase, lang) == expected

    def test_never_zero(self):
        assert _word_count("", Lang.ZH) >= 1


class TestFallbackPath:
    """无 spaCy 模型时的 n-gram 降级路径，无条件运行。"""

    def test_fallback_reports_source(self, monkeypatch: pytest.MonkeyPatch):
        import app.services.guardrails.term_extractor as mod

        monkeypatch.setattr(mod, "_candidates_from_pos", lambda *a, **k: None)
        result = extract_terms(ZH_TEXT, Lang.ZH)
        assert result.skeleton_source == "ngram"

    def test_fallback_finds_key_terms(self, monkeypatch: pytest.MonkeyPatch):
        import app.services.guardrails.term_extractor as mod

        monkeypatch.setattr(mod, "_candidates_from_pos", lambda *a, **k: None)
        terms = terms_of(ZH_TEXT, Lang.ZH)
        assert any("自注意力" in t for t in terms)

    def test_fallback_english(self, monkeypatch: pytest.MonkeyPatch):
        import app.services.guardrails.term_extractor as mod

        monkeypatch.setattr(mod, "_candidates_from_pos", lambda *a, **k: None)
        result = extract_terms(EN_TEXT, Lang.EN)
        assert result.skeleton_source == "ngram"
        assert result.candidates

    def test_fallback_no_newline_artifacts(self, monkeypatch: pytest.MonkeyPatch):
        import app.services.guardrails.term_extractor as mod

        monkeypatch.setattr(mod, "_candidates_from_pos", lambda *a, **k: None)
        multiline = "多头自注意力机制很重要。\n多头自注意力机制再次出现。"
        for term in terms_of(multiline, Lang.ZH):
            assert "\n" not in term


class TestGeneralBehaviour:
    def test_empty_text(self):
        result = extract_terms("", Lang.ZH)
        assert result.candidates == []

    def test_mask_tokens_not_extracted_as_terms(self):
        """`{{CITE_1}}` 里的 CITE 不能被当成术语。"""
        text = "见 {{CITE_1}} 与 {{CITE_2}}，另见 {{MATH_1}} 与 {{MATH_2}}。" * 3
        for term in terms_of(text, Lang.ZH):
            assert "CITE" not in term
            assert "MATH" not in term
            assert "{{" not in term

    def test_top_k_respected(self):
        result = extract_terms(ZH_TEXT, Lang.ZH, top_k=2)
        assert len(result.candidates) <= 2

    def test_candidates_sorted_by_score_desc(self):
        scores = [c.score for c in extract_terms(ZH_TEXT, Lang.ZH).candidates]
        assert scores == sorted(scores, reverse=True)

    def test_singleton_frequency_filtered(self):
        """只出现一次的候选默认不入选（term_min_frequency = 2）。"""
        result = extract_terms("一个只出现一次的特殊名词短语。", Lang.ZH)
        assert result.candidates == []

    def test_all_candidates_have_positive_score(self):
        for candidate in extract_terms(ZH_TEXT, Lang.ZH).candidates:
            assert candidate.score > 0
            assert candidate.frequency >= 2

    def test_no_newline_in_any_candidate(self):
        for candidate in extract_terms(ZH_TEXT, Lang.ZH).candidates:
            assert "\n" not in candidate.term


@pytest.mark.needs_spacy_model
class TestWithRealParser:
    """需要 spaCy 模型：验证词性过滤路径抽出的是完整术语而非碎片。"""

    def _require(self, lang: Lang) -> None:
        if not get_pipeline(lang).available:
            pytest.skip(f"{lang.value} spaCy 模型未安装")

    def test_source_is_pos(self):
        self._require(Lang.ZH)
        assert extract_terms(ZH_TEXT, Lang.ZH).skeleton_source == "pos"

    def test_chinese_full_term_ranked_first(self):
        """「多头自注意力机制」应胜过其子串 —— C-Value 的核心作用。"""
        self._require(Lang.ZH)
        terms = terms_of(ZH_TEXT, Lang.ZH)
        assert terms, "未抽出任何候选"
        assert "多头自注意力机制" in terms[:3]

    def test_chinese_connector_zi_does_not_break_term(self):
        """「自」被 spaCy 标为 ADP，若按词性硬切会把术语腰断成「注意力机制」。"""
        self._require(Lang.ZH)
        assert any("多头自注意力" in t for t in terms_of(ZH_TEXT, Lang.ZH))

    def test_english_hyphenated_term_assembled(self):
        """spaCy 把 multi-head self-attention 切成 6 个 token；
        连接符若计入词数预算，完整术语永远拼不出来。"""
        self._require(Lang.EN)
        assert "multi-head self-attention" in terms_of(EN_TEXT, Lang.EN)

    def test_english_term_not_space_joined(self):
        """按 token 拼接会产出 'self attention'，原文切片才保真。"""
        self._require(Lang.EN)
        terms = terms_of(EN_TEXT, Lang.EN)
        assert "multi head self attention" not in terms
        assert "self attention" not in terms

    def test_verbs_and_adverbs_not_terms(self):
        """「显著提升」是套话而非术语，正是 Track C 要标记的对象。"""
        self._require(Lang.ZH)
        text = "该方法显著提升了性能。该方法显著提升了效果。显著提升十分明显。"
        for term in terms_of(text, Lang.ZH):
            assert term != "显著提升"

    def test_nested_term_is_discounted(self):
        """总作为更长术语一部分出现的子串，得分应低于完整术语。"""
        self._require(Lang.ZH)
        by_term = {c.term: c for c in extract_terms(ZH_TEXT, Lang.ZH).candidates}
        full = by_term.get("多头自注意力机制")
        nested = by_term.get("注意力机制")
        if full is None or nested is None:
            pytest.skip("该语料未同时抽出完整术语与其子串")
        assert nested.nested_in > 0
        assert full.score > nested.score

    def test_acronym_extracted(self):
        self._require(Lang.ZH)
        assert "RMSE" in terms_of(ZH_TEXT, Lang.ZH)
