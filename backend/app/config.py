"""Pydantic Settings 配置管理。

所有可调阈值集中在此，便于用真实语料重标后一处修改。环境变量命名与
仓库根目录的 .env.example 保持一致。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.diagnosis import EngineMode

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 诊断引擎 ----
    diag_engine_mode: EngineMode = EngineMode.DUAL_TRACK
    local_prob_model_path: Path = Path("./models/qwen2.5-1.5b-onnx-int8")
    device: str = "cpu"

    # ---- 大模型协同 ----
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model_name: str = "qwen2.5:14b-instruct"

    # ---- 存储 ----
    database_url: str = "sqlite+aiosqlite:///./data/scholar_refine.db"

    # ---- 资源路径 ----
    dictionaries_dir: Path = _REPO_ROOT / "dictionaries"

    # ---- Track B 阈值 ----
    # 两条骨架路径的相似度分布不同，必须各自标定，不可共用一个阈值：
    # 词性近似骨架把实词压成 X、只留虚词与标点，序列字符重复度天然更高，
    # SequenceMatcher 比值被系统性抬高。共用 0.78 会让同一篇稿子在
    # 「装了 spaCy 模型」与「没装」的环境下得到不同的风险等级。
    parallel_similarity_threshold: float = Field(
        default=0.78, ge=0, le=1, description="真实依存骨架（dep）的排比判定阈值"
    )
    parallel_similarity_threshold_pos: float = Field(
        default=0.90,
        ge=0,
        le=1,
        description="词性近似骨架（pos 降级）的排比判定阈值，待真实语料重标",
    )
    burstiness_human_floor: float = Field(
        default=0.30, description="人类写作句长 CV 典型下界，高于此不罚分"
    )
    burstiness_ai_ceiling: float = Field(
        default=0.18, description="典型 AI 句长 CV 上界，低于此满分罚"
    )
    min_sentences_for_burstiness: int = Field(default=3, ge=1)

    # ---- Track C 阈值 ----
    fluff_abstract_min_count: int = Field(
        default=2, ge=1, description="单句抽象词数达此值且无具象锚点即触发 LOW_INFO_DENSITY"
    )

    # ---- 术语预提取（C-Value）----
    term_max_words: int = Field(
        default=5, ge=1, description="候选术语的最大词数（中文按 2 字/词近似）"
    )
    term_min_frequency: int = Field(
        default=2, ge=1, description="进入 C-Value 计算的最低出现次数，低频候选噪声大"
    )
    term_min_score: float = Field(default=1.0, ge=0, description="候选术语的 C-Value 下限")
    term_top_k: int = Field(default=30, ge=1, description="推送给前端的候选术语上限")

    # ---- 聚合权重 ----
    weight_dual_syntactic: float = 0.45
    weight_dual_lexicon: float = 0.55
    weight_tri_curvature: float = 0.35
    weight_tri_syntactic: float = 0.30
    weight_tri_lexicon: float = 0.35

    # ---- spaCy ----
    spacy_model_zh: str = "zh_core_web_sm"
    spacy_model_en: str = "en_core_web_sm"
    spacy_require_model: bool = Field(
        default=False,
        description="True 时模型缺失直接报错；False 时降级为词性近似骨架（CI 默认）",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
