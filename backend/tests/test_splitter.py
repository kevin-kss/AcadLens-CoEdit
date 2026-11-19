"""分句器单测。

这个模块的 bug 会污染下游全部诊断结果（句长腰斩 -> CV 失真 -> 风险分错误），
因此覆盖重点是**否决规则**：学术文本里大量句号不是句子边界。
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.schemas.diagnosis import Lang
from app.services.parser.splitter import split_paragraphs, split_sentences


def texts(raw: str, lang: Lang = Lang.ZH) -> list[str]:
    return [s.text for s in split_sentences(raw, lang)]


class TestChineseBasics:
    def test_cjk_enders(self):
        assert texts("模型收敛了。实验结果良好！还有疑问？") == [
            "模型收敛了。",
            "实验结果良好！",
            "还有疑问？",
        ]

    def test_semicolon_and_ellipsis(self):
        assert texts("第一点如下；第二点稍后……") == ["第一点如下；", "第二点稍后……"]

    def test_trailing_quote_stays_with_sentence(self):
        assert texts("他说「这是关键。」然后离开了。") == [
            "他说「这是关键。」",
            "然后离开了。",
        ]

    def test_no_ender_tail_is_kept(self):
        assert texts("摘要末行没有句号") == ["摘要末行没有句号"]

    def test_empty_and_whitespace(self):
        assert split_sentences("", Lang.ZH) == []
        assert split_sentences("   \n  ", Lang.ZH) == []


class TestEnglishAbbreviations:
    """文档 M1 明确要求跳过的缩写。"""

    @pytest.mark.parametrize(
        "raw",
        [
            "As shown in Fig. 1 the loss decreases monotonically.",
            "Results appear in Table. 3 below.",
            "We follow the setup of Sec. 4 closely.",
            "See Eq. 7 for the derivation.",
            "Refer to Ref. 12 for details.",
            "The data come from Vol. 3 of the corpus.",
            "Dr. Smith supervised the annotation.",
            "Prof. Lee proposed the metric.",
            "Compare with cf. the baseline configuration.",
            "We report approx. 40 ms per sentence.",
        ],
    )
    def test_hard_abbreviations_not_split(self, raw: str):
        assert texts(raw, Lang.EN) == [raw]

    @pytest.mark.parametrize(
        "raw",
        [
            "We use a small batch, e.g. 16 samples per step.",
            "The model is autoregressive, i.e. it predicts one token at a time.",
            "The corpus covers U.S. federal filings only.",
            "She holds a Ph.D. in computational linguistics.",
            "Runs started at 9 a.m. on three machines.",
        ],
    )
    def test_dotted_abbreviations_not_split(self, raw: str):
        assert texts(raw, Lang.EN) == [raw]

    def test_personal_initial_not_split(self):
        raw = "The method of J. Smith remains the strongest baseline."
        assert texts(raw, Lang.EN) == [raw]

    def test_multiple_initials_not_split(self):
        raw = "Credit goes to A. B. Chen for the annotation protocol."
        assert texts(raw, Lang.EN) == [raw]


class TestDecimalsAndNumbers:
    def test_decimal_not_split(self):
        raw = "The constant equals 3.14 in all runs."
        assert texts(raw, Lang.EN) == [raw]

    def test_section_number_not_split(self):
        raw = "Details live in subsection 4.2 of the appendix."
        assert texts(raw, Lang.EN) == [raw]

    def test_version_string_not_split(self):
        raw = "We ran spaCy 3.8.0 with the small Chinese model."
        assert texts(raw, Lang.EN) == [raw]

    def test_url_not_split(self):
        raw = "Code is hosted at github.com/example/repo for review."
        assert texts(raw, Lang.EN) == [raw]


class TestEtAlSoftRule:
    """`et al.` / `etc.` 既可能句中也可能句末 —— 看后续是否为新句开头。"""

    def test_et_al_followed_by_citation_stays(self):
        raw = "The protocol of Smith et al. (2024) was reproduced exactly."
        assert texts(raw, Lang.EN) == [raw]

    def test_et_al_followed_by_new_sentence_splits(self):
        raw = "This mirrors Smith et al. The remaining gap is unexplained."
        assert texts(raw, Lang.EN) == [
            "This mirrors Smith et al.",
            "The remaining gap is unexplained.",
        ]

    def test_etc_mid_sentence(self):
        raw = "We tune depth, width, etc. using random search."
        assert texts(raw, Lang.EN) == [raw]

    def test_etc_before_chinese_sentence_splits(self):
        raw = "We tune depth, width, etc. 随后进行消融实验。"
        assert texts(raw, Lang.EN) == [
            "We tune depth, width, etc.",
            "随后进行消融实验。",
        ]


class TestMaskTokenSafety:
    def test_no_split_inside_mask_token(self):
        raw = "该结论见 {{CITE_1}}，与 {{MATH_2}} 的推导一致。"
        assert texts(raw) == [raw]

    def test_citation_at_sentence_end(self):
        raw = "该方法由前人提出 {{CITE_1}}。我们在此基础上扩展。"
        assert texts(raw) == [
            "该方法由前人提出 {{CITE_1}}。",
            "我们在此基础上扩展。",
        ]


class TestOffsets:
    def test_offsets_slice_back_exactly(self):
        raw = "第一句在此。  第二句稍长一些，包含逗号。第三句结束。"
        for span in split_sentences(raw, Lang.ZH):
            assert raw[span.start_char : span.end_char] == span.text

    def test_offsets_are_monotonic(self):
        raw = "Alpha holds. Beta follows. Gamma concludes the argument."
        spans = split_sentences(raw, Lang.EN)
        assert [s.index for s in spans] == [0, 1, 2]
        for prev, cur in pairwise(spans):
            assert prev.end_char <= cur.start_char

    def test_english_offsets_slice_back(self):
        raw = "See Fig. 1 for the curve. Accuracy reaches 94.2% at epoch 30."
        for span in split_sentences(raw, Lang.EN):
            assert raw[span.start_char : span.end_char] == span.text


class TestMixedAndParagraphs:
    def test_mixed_zh_en(self):
        raw = "我们采用 Transformer 架构。As shown in Fig. 2, 准确率提升了 4.2%。"
        assert texts(raw) == [
            "我们采用 Transformer 架构。",
            "As shown in Fig. 2, 准确率提升了 4.2%。",
        ]

    def test_consecutive_enders_collapse(self):
        assert texts("真的吗？！确实如此。") == ["真的吗？！", "确实如此。"]

    def test_split_paragraphs_drops_blank(self):
        raw = "第一段内容。\n\n第二段内容。\n\n\n  \n第三段内容。"
        assert split_paragraphs(raw) == ["第一段内容。", "第二段内容。", "第三段内容。"]
