"""词库与规则资源加载器（支持热加载）。

词库是业务资产，会被反复增删条目。这里的约定：
- `.txt` 每行一条，`#` 开头为注释，空行忽略，自动去重并剔除首尾空白；
- `.json` 帽句规则在加载时即编译正则，坏规则**不静默忽略**而是抛错 ——
  一条写错的正则会让该类问题在全库范围内漏检，比启动失败难发现得多。

加载结果按 (路径, mtime) 缓存，文件改动后下次调用自动重编译，无需重启进程。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import get_settings
from app.schemas.diagnosis import Lang

PatternPosition = Literal["head", "tail", "any"]


class LexiconError(RuntimeError):
    """词库文件缺失或规则非法。"""


@dataclass(frozen=True)
class HatBootRule:
    rule_id: str
    position: PatternPosition
    pattern: re.Pattern[str]
    severity: int
    message: str


@dataclass(frozen=True)
class LangLexicon:
    lang: Lang
    connectors: tuple[str, ...]
    fluff: tuple[str, ...]
    hat_boot_rules: tuple[HatBootRule, ...]
    stopwords: frozenset[str]


def _read_wordlist(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise LexiconError(f"词库文件缺失: {path}")
    seen: dict[str, None] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        seen.setdefault(line, None)
    if not seen:
        raise LexiconError(f"词库为空: {path}")
    return tuple(seen)


def _read_hat_boot(path: Path) -> tuple[HatBootRule, ...]:
    if not path.is_file():
        raise LexiconError(f"帽句规则文件缺失: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LexiconError(f"帽句规则 JSON 解析失败 {path}: {exc}") from exc

    rules: list[HatBootRule] = []
    seen_ids: set[str] = set()
    for item in payload.get("patterns", []):
        rule_id = item.get("id")
        if not rule_id:
            raise LexiconError(f"{path}: 存在缺少 id 的规则")
        if rule_id in seen_ids:
            raise LexiconError(f"{path}: 规则 id 重复 -> {rule_id}")
        seen_ids.add(rule_id)

        position = item.get("position", "any")
        if position not in ("head", "tail", "any"):
            raise LexiconError(f"{path}:{rule_id} position 非法 -> {position}")

        try:
            compiled = re.compile(item["regex"], re.IGNORECASE)
        except (KeyError, re.error) as exc:
            raise LexiconError(f"{path}:{rule_id} 正则非法 -> {exc}") from exc

        rules.append(
            HatBootRule(
                rule_id=rule_id,
                position=position,
                pattern=compiled,
                severity=int(item.get("severity", 2)),
                message=item.get("message", "疑似帽句或升华套话"),
            )
        )
    if not rules:
        raise LexiconError(f"{path}: 规则列表为空")
    return tuple(rules)


_CACHE: dict[tuple[Lang, tuple[float, ...]], LangLexicon] = {}


def _dir_for(lang: Lang) -> Path:
    return get_settings().dictionaries_dir / lang.value


def _mtimes(paths: list[Path]) -> tuple[float, ...]:
    return tuple(p.stat().st_mtime if p.is_file() else 0.0 for p in paths)


def load_lexicon(lang: Lang | str) -> LangLexicon:
    """加载某语种的全部词库资源，带 mtime 失效的进程内缓存。"""
    lang_enum = Lang(lang)
    base = _dir_for(lang_enum)
    connector_path = base / "ai_connectors.txt"
    fluff_path = base / "abstract_fluff.txt"
    hat_boot_path = base / "hat_boot_patterns.json"
    stopword_path = base / "academic_stopwords.txt"

    watched = [connector_path, fluff_path, hat_boot_path, stopword_path]
    cache_key = (lang_enum, _mtimes(watched))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    stopwords: frozenset[str] = frozenset()
    if stopword_path.is_file():
        stopwords = frozenset(_read_wordlist(stopword_path))

    lexicon = LangLexicon(
        lang=lang_enum,
        connectors=_read_wordlist(connector_path),
        fluff=_read_wordlist(fluff_path),
        hat_boot_rules=_read_hat_boot(hat_boot_path),
        stopwords=stopwords,
    )
    _CACHE.clear()  # 只保留最新一版，避免旧 mtime 条目堆积
    _CACHE[cache_key] = lexicon
    return lexicon


def clear_cache() -> None:
    """测试与词库热更新后手动失效用。"""
    _CACHE.clear()
