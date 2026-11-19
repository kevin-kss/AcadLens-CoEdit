"""只读审查发现的 6 处问题的回归测试。

每个类对应一条，注释写明「错在哪、后果是什么」，避免日后被当作多余约束删掉。
这些问题的共同特征是不抛异常：分数照算、接口不变，只是算错或定位错位。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.schemas.diagnosis import IssueType, Lang
from app.services.diagnostics import lexicon_loader
from app.services.diagnostics.lexicon_loader import _dir_for, clear_cache, load_lexicon
from app.services.diagnostics.text_metrics import effective_length
from app.services.diagnostics.track_c_lexicon import (
    _lower_preserving_length,
    _scan_phrases,
    diagnose_track_c,
)
from app.services.parser.masker import mask_academic_text, unmask_academic_text
from app.services.parser.splitter import split_sentences


class TestUrlDoesNotSwallowSentenceEnd:
    """问题 1：URL 正则吞掉句末标点，分句器因掩码 token 跳过而无法识别句界，
    相邻两句合并成一句，连带污染句长、句法与段落风险统计。"""

    @pytest.mark.parametrize(
        ("raw", "want_sentences"),
        [
            ("See https://example.com. Results follow.", 2),
            ("URL 在句末 https://example.com。后续见下节。", 2),
            ("见 doi:10.1145/3372297.3417231 原文。后续见下节。", 2),
            ("Refer to 10.1145/3372297.3417231. Then compare.", 2),
            # URL 在句中时不应被截断
            ("Code at https://github.com/a/b for review.", 1),
            ("Visit https://example.com/path?q=1&r=2 now.", 1),
            ("数据集见 https://example.com，随后分析。", 1),
            ("Trailing slash https://example.com/ okay.", 1),
            ("See (https://example.com) for details.", 1),
        ],
    )
    def test_sentence_count_preserved(self, raw: str, want_sentences: int):
        masked = mask_academic_text(raw)
        assert len(split_sentences(masked.masked, Lang.EN)) == want_sentences

    @pytest.mark.parametrize(
        "raw",
        [
            "See https://example.com. Results follow.",
            "URL 在句末 https://example.com。后续见下节。",
            "见 doi:10.1145/3372297.3417231 原文。",
            "Visit https://example.com/path?q=1&r=2 now.",
            "数据集见 https://example.com，随后分析。",
        ],
    )
    def test_roundtrip_still_lossless(self, raw: str):
        """修复不得牺牲无损还原 —— 这是硬验收指标。"""
        masked = mask_academic_text(raw)
        assert unmask_academic_text(masked.masked, masked.registry) == raw

    def test_url_token_excludes_trailing_punctuation(self):
        masked = mask_academic_text("See https://example.com. Results follow.")
        assert list(masked.registry.mapping.values()) == ["https://example.com"]

    def test_url_does_not_swallow_following_chinese(self):
        """\\S* 贪婪匹配会把 URL 后面的中文正文一起吞掉。"""
        masked = mask_academic_text("URL 在句末 https://example.com。后续见下节。")
        assert list(masked.registry.mapping.values()) == ["https://example.com"]


class TestThresholdPerSkeletonSource:
    """问题 2：dep 与 pos 两条骨架路径共用 0.78 阈值，使同一段文本在装/不装
    spaCy 模型的环境下落进不同风险等级。"""

    def test_thresholds_are_distinct(self):
        from app.services.diagnostics.track_b_syntactic import threshold_for_source

        assert threshold_for_source("pos") != threshold_for_source("dep")

    def test_pos_threshold_is_stricter(self):
        """词性近似骨架把实词压成 X，字符重复度天然更高，阈值须更严。"""
        from app.services.diagnostics.track_b_syntactic import threshold_for_source

        assert threshold_for_source("pos") > threshold_for_source("dep")

    def test_source_dispatches_threshold(self):
        from app.services.diagnostics.track_b_syntactic import (
            detect_syntactic_parallelism,
        )

        skeletons = ["X-的-X-X-了-X", "X-的-X-X-X-了"]
        assert detect_syntactic_parallelism(skeletons, source="dep") == [0, 1]
        assert detect_syntactic_parallelism(skeletons, source="pos") == []

    def test_explicit_threshold_still_wins(self):
        from app.services.diagnostics.track_b_syntactic import (
            detect_syntactic_parallelism,
        )

        skeletons = ["A-B-C-D-E", "A-B-C-D-F"]
        assert detect_syntactic_parallelism(skeletons, threshold=0.5, source="pos") == [0, 1]


class TestDensityDenominatorExcludesMasks:
    """问题 3：密度分母用带掩码的原文长度，`{{CITE_1}}` 长 10~12 字符，
    多加几条引文就能稀释密度、凭空降低风险分，而正文一字未改。"""

    def test_mask_tokens_do_not_dilute_score(self):
        base = "此外，我们分析了结果。"
        padded = base + " {{CITE_1}} {{CITE_2}} {{CITE_3}} {{MATH_1}}"
        base_score, _, _ = diagnose_track_c(split_sentences(base, Lang.ZH), Lang.ZH)
        padded_score, _, _ = diagnose_track_c(split_sentences(padded, Lang.ZH), Lang.ZH)
        assert base_score == pytest.approx(padded_score)

    def test_english_mask_tokens_do_not_dilute(self):
        base = "Moreover, we analyzed the results."
        padded = base + " {{CITE_1}} {{CITE_2}} {{MATH_1}}"
        base_score, _, _ = diagnose_track_c(split_sentences(base, Lang.EN), Lang.EN)
        padded_score, _, _ = diagnose_track_c(split_sentences(padded, Lang.EN), Lang.EN)
        assert base_score == pytest.approx(padded_score)


class TestCaseFoldingPreservesOffsets:
    """问题 4：在 text.lower() 后计算位置却用这些位置切原文。Unicode 大小写
    展开会改变长度（'İ'.lower() 为两个码点），命中偏移整体错位，
    前端下划线画错位置。"""

    @pytest.mark.parametrize(
        "text",
        [
            "İ Moreover, results hold.",  # İ 小写展开为 2 码点
            "İİİ Furthermore, it holds.",
            "Straße. Moreover, results hold.",
            "ﬁle Moreover, results hold.",  # ﬁ 连字
            "Moreover, results hold.",
            "FURTHERMORE, results hold.",
        ],
    )
    def test_lower_preserves_length(self, text: str):
        assert len(_lower_preserving_length(text)) == len(text)

    @pytest.mark.parametrize(
        "text",
        [
            "İ Moreover, results hold.",
            "İİİ Furthermore, it holds.",
            "Straße. Moreover, results hold.",
            "Moreover, results hold.",
            "FURTHERMORE, results hold.",
        ],
    )
    def test_hit_offsets_slice_back_correctly(self, text: str):
        lexicon = load_lexicon(Lang.EN)
        hits = _scan_phrases(text, lexicon.connectors, Lang.EN, "connector")
        assert hits, f"未命中任何套话: {text!r}"
        for start, end, word in hits:
            assert text[start:end].lower() == word.lower()

    def test_issue_offsets_match_matched_text(self):
        text = "İ Moreover, results hold."
        _, _, issues = diagnose_track_c(split_sentences(text, Lang.EN), Lang.EN)
        assert issues
        for issue in issues:
            assert text[issue.start_char : issue.end_char] == issue.matched_text

    def test_matched_text_is_not_truncated(self):
        """错位的具体表现：返回 'oreover,' 而非 'Moreover'。"""
        text = "İ Moreover, results hold."
        _, _, issues = diagnose_track_c(split_sentences(text, Lang.EN), Lang.EN)
        connectors = [i for i in issues if i.issue_type is IssueType.AI_CONNECTOR]
        assert any(i.matched_text == "Moreover" for i in connectors)


class TestEffectiveLengthCountsTermsAsUnits:
    """问题 5：`\\w` 不只匹配中文字符，英文术语、数字与下划线被逐字符计入，
    "模型使用Transformer v2。" 得 17 而非按字的 6，使中英混排论文的句长 CV
    系统性偏离代码注释声明的「中文按字计」。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("模型使用Transformer v2。", 6),  # 4 字 + Transformer + v2
            ("模型使用它。", 5),
            ("准确率为76.3%。", 5),  # 4 字 + 76.3
            ("The model converged quickly.", 4),
            ("我们在 ImageNet-1k 上训练 ResNet-50。", 8),  # 6 字 + 2 术语
            ("", 0),
        ],
    )
    def test_latin_runs_count_as_one_unit(self, text: str, expected: int):
        assert effective_length(text, Lang.ZH) == expected

    def test_long_term_does_not_inflate_length(self):
        """同样句式，仅术语变长，句长不应随之膨胀。"""
        short = effective_length("模型使用它。", Lang.ZH)
        long = effective_length("模型使用Transformer。", Lang.ZH)
        assert abs(short - long) <= 1

    def test_mask_tokens_excluded(self):
        assert effective_length("该结论见 {{CITE_1}}。", Lang.ZH) == effective_length(
            "该结论见 。", Lang.ZH
        )

    def test_underscore_not_counted_separately(self):
        """变+量+为 = 3 字，batch_size 与 256 各计 1 个单位，共 5。

        下划线若被当作分隔符，batch_size 会被拆成两个单位而得 6。
        """
        assert effective_length("变量 batch_size 为 256。", Lang.ZH) == 5


class TestLexiconCacheKeepsBothLanguages:
    """问题 6：缓存以 (语种, mtime) 为 key 且每次加载整表 clear，中英段落
    交替诊断时两套词库互相挤出缓存、反复读盘并重建 Aho-Corasick 自动机。"""

    def test_both_languages_cached_simultaneously(self):
        clear_cache()
        zh_first = load_lexicon(Lang.ZH)
        en_first = load_lexicon(Lang.EN)
        assert load_lexicon(Lang.ZH) is zh_first
        assert load_lexicon(Lang.EN) is en_first

    def test_alternating_access_hits_cache(self):
        clear_cache()
        zh_first = load_lexicon(Lang.ZH)
        en_first = load_lexicon(Lang.EN)
        for _ in range(3):
            assert load_lexicon(Lang.ZH) is zh_first
            assert load_lexicon(Lang.EN) is en_first

    def test_cache_holds_one_entry_per_language(self):
        clear_cache()
        load_lexicon(Lang.ZH)
        load_lexicon(Lang.EN)
        assert len(lexicon_loader._CACHE) == 2

    def test_hot_reload_still_invalidates_only_that_language(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """热加载必须仍然生效，且不应连带失效另一语种。"""
        root = tmp_path / "dictionaries"
        for lang in ("zh", "en"):
            src = _dir_for(Lang(lang))
            dst = root / lang
            dst.mkdir(parents=True)
            for name in (
                "ai_connectors.txt",
                "abstract_fluff.txt",
                "hat_boot_patterns.json",
                "academic_stopwords.txt",
            ):
                (dst / name).write_bytes((src / name).read_bytes())

        settings = Settings(dictionaries_dir=root)
        monkeypatch.setattr(lexicon_loader, "get_settings", lambda: settings)
        clear_cache()

        zh_first = load_lexicon(Lang.ZH)
        en_first = load_lexicon(Lang.EN)

        path = root / "zh" / "ai_connectors.txt"
        path.write_text(path.read_text(encoding="utf-8") + "\n新增套话条目\n", encoding="utf-8")
        future = path.stat().st_mtime + 10
        os.utime(path, (future, future))

        zh_second = load_lexicon(Lang.ZH)
        assert zh_second is not zh_first
        assert "新增套话条目" in zh_second.connectors
        assert load_lexicon(Lang.EN) is en_first


class TestRepoDictionariesUnchanged:
    """守护：上述验证脚本曾临时写入仓库词库，确保已恢复。"""

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_no_test_artifacts_left_in_dictionaries(self, lang: Lang):
        clear_cache()
        connectors = load_lexicon(lang).connectors
        for leaked in ("测试新增条目", "新增套话条目", "新增套话"):
            assert leaked not in connectors

    def test_dictionaries_dir_is_repo_root(self):
        assert (get_settings().dictionaries_dir / "zh" / "ai_connectors.txt").is_file()
