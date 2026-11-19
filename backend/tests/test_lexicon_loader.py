"""词库加载器单测。

重点：坏规则必须**启动即报错**而非静默忽略 —— 一条写错的正则会让该类问题
在全库范围内漏检，比启动失败难发现得多。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.schemas.diagnosis import Lang
from app.services.diagnostics import lexicon_loader
from app.services.diagnostics.lexicon_loader import (
    LexiconError,
    clear_cache,
    load_lexicon,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


def _write_lexicon_dir(root: Path, lang: str, *, hat_boot: object | None = None) -> Path:
    base = root / lang
    base.mkdir(parents=True, exist_ok=True)
    (base / "ai_connectors.txt").write_text("此外\n值得注意的是\n", encoding="utf-8")
    (base / "abstract_fluff.txt").write_text("多维度\n赋能\n", encoding="utf-8")
    payload = (
        hat_boot
        if hat_boot is not None
        else {
            "patterns": [
                {
                    "id": "t_head",
                    "position": "head",
                    "regex": "^在当今",
                    "severity": 3,
                    "message": "测试帽句",
                }
            ]
        }
    )
    (base / "hat_boot_patterns.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return base


@pytest.fixture
def temp_dicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把 settings.dictionaries_dir 指向临时目录。"""

    def _factory(**kwargs: object) -> Path:
        root = tmp_path / "dictionaries"
        _write_lexicon_dir(root, "zh", **kwargs)
        settings = Settings(dictionaries_dir=root)
        monkeypatch.setattr(lexicon_loader, "get_settings", lambda: settings)
        clear_cache()
        return root

    return _factory


class TestRealDictionaries:
    """仓库内置词库必须可加载 —— 这是 M1 的交付物。"""

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_loads(self, lang: Lang):
        lex = load_lexicon(lang)
        assert lex.lang is lang
        assert lex.connectors
        assert lex.fluff
        assert lex.hat_boot_rules

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_milestone_size_requirements(self, lang: Lang):
        """M1 要求：中英文套话词典各 150+ 条，帽句规则 30+ 条。"""
        lex = load_lexicon(lang)
        assert len(lex.connectors) >= 150, f"{lang} 套话词条不足: {len(lex.connectors)}"
        assert len(lex.hat_boot_rules) >= 30, f"{lang} 帽句规则不足: {len(lex.hat_boot_rules)}"

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_stopwords_present(self, lang: Lang):
        assert load_lexicon(lang).stopwords

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_all_regexes_compile_and_are_usable(self, lang: Lang):
        for rule in load_lexicon(lang).hat_boot_rules:
            assert rule.pattern.search("") is None or True  # 不抛异常即可
            assert rule.position in ("head", "tail", "any")
            assert rule.message

    @pytest.mark.parametrize("lang", [Lang.ZH, Lang.EN])
    def test_rule_ids_unique(self, lang: Lang):
        ids = [r.rule_id for r in load_lexicon(lang).hat_boot_rules]
        assert len(ids) == len(set(ids))

    def test_accepts_string_lang(self):
        assert load_lexicon("zh").lang is Lang.ZH

    def test_cache_returns_same_object(self):
        assert load_lexicon(Lang.ZH) is load_lexicon(Lang.ZH)


class TestWordlistParsing:
    def test_comments_and_blanks_ignored(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "dictionaries"
        base = _write_lexicon_dir(root, "zh")
        (base / "ai_connectors.txt").write_text(
            "# 注释行\n\n此外\n  值得注意的是  \n# 又一条注释\n此外\n",
            encoding="utf-8",
        )
        settings = Settings(dictionaries_dir=root)
        monkeypatch.setattr(lexicon_loader, "get_settings", lambda: settings)
        clear_cache()
        # 去重 + 去首尾空白
        assert load_lexicon(Lang.ZH).connectors == ("此外", "值得注意的是")

    def test_missing_file_raises(self, temp_dicts):
        root = temp_dicts()
        (root / "zh" / "ai_connectors.txt").unlink()
        clear_cache()
        with pytest.raises(LexiconError, match="词库文件缺失"):
            load_lexicon(Lang.ZH)

    def test_empty_file_raises(self, temp_dicts):
        root = temp_dicts()
        (root / "zh" / "abstract_fluff.txt").write_text("# 只有注释\n", encoding="utf-8")
        clear_cache()
        with pytest.raises(LexiconError, match="词库为空"):
            load_lexicon(Lang.ZH)


class TestBadRulesFailLoudly:
    def test_invalid_regex_raises(self, temp_dicts):
        temp_dicts(
            hat_boot={
                "patterns": [
                    {"id": "bad", "position": "any", "regex": "([unclosed", "message": "x"}
                ]
            }
        )
        with pytest.raises(LexiconError, match="正则非法"):
            load_lexicon(Lang.ZH)

    def test_duplicate_id_raises(self, temp_dicts):
        temp_dicts(
            hat_boot={
                "patterns": [
                    {"id": "dup", "position": "any", "regex": "a", "message": "x"},
                    {"id": "dup", "position": "any", "regex": "b", "message": "y"},
                ]
            }
        )
        with pytest.raises(LexiconError, match="规则 id 重复"):
            load_lexicon(Lang.ZH)

    def test_missing_id_raises(self, temp_dicts):
        temp_dicts(hat_boot={"patterns": [{"position": "any", "regex": "a"}]})
        with pytest.raises(LexiconError, match="缺少 id"):
            load_lexicon(Lang.ZH)

    def test_bad_position_raises(self, temp_dicts):
        temp_dicts(
            hat_boot={"patterns": [{"id": "p", "position": "middle", "regex": "a", "message": "x"}]}
        )
        with pytest.raises(LexiconError, match="position 非法"):
            load_lexicon(Lang.ZH)

    def test_empty_pattern_list_raises(self, temp_dicts):
        temp_dicts(hat_boot={"patterns": []})
        with pytest.raises(LexiconError, match="规则列表为空"):
            load_lexicon(Lang.ZH)

    def test_malformed_json_raises(self, temp_dicts):
        root = temp_dicts()
        (root / "zh" / "hat_boot_patterns.json").write_text("{not json", encoding="utf-8")
        clear_cache()
        with pytest.raises(LexiconError, match="JSON 解析失败"):
            load_lexicon(Lang.ZH)


class TestHotReload:
    def test_mtime_change_invalidates_cache(self, temp_dicts):
        """词库是业务资产，改动后下次调用即生效，无需重启进程。"""
        import os

        root = temp_dicts()
        first = load_lexicon(Lang.ZH)
        assert "新增套话" not in first.connectors

        path = root / "zh" / "ai_connectors.txt"
        path.write_text("此外\n值得注意的是\n新增套话\n", encoding="utf-8")
        # 显式推进 mtime，避免同秒写入导致时间戳不变
        future = path.stat().st_mtime + 10
        os.utime(path, (future, future))

        second = load_lexicon(Lang.ZH)
        assert "新增套话" in second.connectors


class TestSettingsDefaults:
    def test_dictionaries_dir_points_at_repo_root(self):
        settings = get_settings()
        assert (settings.dictionaries_dir / "zh" / "ai_connectors.txt").is_file()
        assert (settings.dictionaries_dir / "en" / "ai_connectors.txt").is_file()
