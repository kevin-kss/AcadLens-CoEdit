"""掩码器单测。

无损还原是硬验收指标（公式与引文一致率 100%），因此核心断言是
**往返一致性**：mask -> unmask 必须与原文逐字符相等。
"""

from __future__ import annotations

import pytest

from app.schemas.document import MaskRegistry
from app.services.parser.masker import (
    MASK_TOKEN_RE,
    find_mask_tokens,
    mask_academic_text,
    strip_mask_tokens,
    unmask_academic_text,
    verify_roundtrip,
)


class TestRoundtrip:
    @pytest.mark.parametrize(
        "raw",
        [
            "损失函数定义为 $$L = \\sum_i (y_i - \\hat{y}_i)^2$$，其中 $y_i$ 为真值。",
            "该结论与前人研究一致 [12]，也见 [1-3] 与 [4, 7, 9]。",
            "The protocol follows Smith et al., 2024 as described in (Lee & Park, 2023a).",
            "推导过程见 \\[ \\frac{\\partial L}{\\partial w} = 0 \\] 以及 \\(x > 0\\)。",
            "参见 \\cite{smith2024} 与 \\ref{fig:arch}，公式编号为 \\eqref{eq:3}。",
            "数据集地址 https://example.com/data?v=2 与 doi:10.1145/3372297.3417231。",
            "\\begin{equation} E = mc^2 \\end{equation} 是著名结论。",
            "混合情形：$\\alpha$ 在 [5] 中取 0.3（Chen 等, 2022），详见 \\label{sec:method}。",
            "纯中文段落没有任何公式与引文，应当原样返回。",
            "",
        ],
    )
    def test_roundtrip_is_lossless(self, raw: str):
        assert verify_roundtrip(raw)

    def test_roundtrip_with_many_tokens(self):
        """验收指标：20+ 公式与 50+ 引文的文档必须 100% 还原。"""
        parts = [f"第 {i} 句包含公式 $x_{i}$ 与引文 [{i}]。" for i in range(1, 61)]
        raw = "".join(parts)
        result = mask_academic_text(raw)
        assert unmask_academic_text(result.masked, result.registry) == raw
        assert len(result.registry.mapping) >= 120

    def test_prefix_collision_token_11_vs_token_1(self):
        """{{CITE_11}} 不得被 {{CITE_1}} 的前缀破坏。"""
        raw = " ".join(f"论点 [{i}]。" for i in range(1, 16))
        result = mask_academic_text(raw)
        assert "{{CITE_11}}" in result.masked
        assert unmask_academic_text(result.masked, result.registry) == raw


class TestPatternCoverage:
    def test_display_math_masked_whole(self):
        raw = "公式 $$a + b = c$$ 成立。"
        result = mask_academic_text(raw)
        assert "$$" not in result.masked
        assert list(result.registry.mapping.values()) == ["$$a + b = c$$"]

    def test_inline_math_masked(self):
        result = mask_academic_text("当 $p < 0.05$ 时显著。")
        assert "{{MATH_1}}" in result.masked

    def test_bracket_citations(self):
        result = mask_academic_text("见 [12]、[1-3]、[4, 7]。")
        assert len(result.registry.mapping) == 3

    def test_author_year_citation(self):
        result = mask_academic_text("该方法 (Smith et al., 2024) 表现最好。")
        assert "{{CITE_1}}" in result.masked

    def test_chinese_author_year_citation(self):
        result = mask_academic_text("该方法（李雷等, 2024）被广泛采用。")
        assert any("2024" in v for v in result.registry.mapping.values())

    def test_latex_cite_command(self):
        result = mask_academic_text("如 \\citep{wang2023} 所述。")
        assert "{{LATEX_CMD_1}}" in result.masked

    def test_url_masked_before_citation(self):
        """doi 中的数字不应被引文规则截断。"""
        raw = "见 doi:10.1145/3372297.3417231 原文。"
        result = mask_academic_text(raw)
        assert unmask_academic_text(result.masked, result.registry) == raw
        assert len(result.registry.mapping) == 1

    def test_equation_environment(self):
        raw = "\\begin{align} x &= 1 \\\\ y &= 2 \\end{align}"
        result = mask_academic_text(raw)
        assert result.masked.strip() == "{{MATH_1}}"


class TestRegistrySharing:
    def test_shared_registry_counter_increments(self):
        """同一文档多段共享全局映射表，编号不复用。"""
        reg = MaskRegistry()
        first = mask_academic_text("第一段见 [1]。", reg)
        second = mask_academic_text("第二段见 [2]。", reg)
        assert "{{CITE_1}}" in first.masked
        assert "{{CITE_2}}" in second.masked
        assert len(reg.mapping) == 2

    def test_registry_tokens_property(self):
        reg = MaskRegistry()
        mask_academic_text("见 [1] 与 $x$。", reg)
        assert set(reg.tokens) == set(reg.mapping)


class TestHelpers:
    def test_strip_mask_tokens(self):
        assert strip_mask_tokens("前 {{CITE_1}} 后 {{MATH_2}}") == "前  后 "

    def test_find_mask_tokens_order(self):
        found = find_mask_tokens("{{CITE_1}} 中间 {{MATH_2}} 末尾 {{CITE_3}}")
        assert found == ["{{CITE_1}}", "{{MATH_2}}", "{{CITE_3}}"]

    def test_mask_token_re_matches_generated_format(self):
        reg = MaskRegistry()
        token = reg.register("CITE", "[9]")
        assert MASK_TOKEN_RE.fullmatch(token)

    def test_unmask_is_noop_without_tokens(self):
        reg = MaskRegistry()
        assert unmask_academic_text("普通文本", reg) == "普通文本"
