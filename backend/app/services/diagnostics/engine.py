"""三轨调度与加权评分聚合器。

权重（见技术文档 4.2.4）：
- 完整三轨：Risk = 0.35*curvature + 0.30*syntactic + 0.35*lexicon
- 轻量双轨：Risk = 0.45*syntactic + 0.55*lexicon

Track A 在 M4 接入；当前 tri_track 模式下 curvature 为 None 时自动退回双轨权重，
避免配置与实际能力不一致时静默算出错误分数。

注意：本模块是 **CPU 密集的纯函数**，不要写成 async def —— GIL 会让并发诊断
请求互相排队，且阻塞事件循环。并发由上层的 ProcessPoolExecutor 承担。
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas.diagnosis import (
    DocumentRiskSummary,
    EngineMode,
    IssueType,
    Lang,
    ParagraphDiagnosis,
    RiskLevel,
    SentenceMetric,
    TrackScores,
)
from app.schemas.document import ParsedParagraph, SentenceSpan
from app.services.diagnostics.track_b_syntactic import (
    diagnose_track_b,
    effective_length,
)
from app.services.diagnostics.track_c_lexicon import TrackCSentenceResult, diagnose_track_c
from app.services.parser.masker import strip_mask_tokens
from app.services.parser.splitter import split_sentences


def _sentence_risk(
    fluff_ratio: float,
    is_parallel: bool,
    has_hat_boot: bool,
    has_low_info: bool,
    connector_hits: int,
    clean_len: int,
) -> float:
    """句级风险分：用于编辑器内逐句着色，与段落分独立计算。

    段落分回答「这段整体像不像机器写的」，句级分回答「先改哪句」，
    两者量纲不同，不能互相推导。
    """
    score = 0.0
    # 连接词按密度计，短句里一个连接词的占比远高于长句
    unit = max(clean_len, 10)
    score += min(40.0, connector_hits / unit * 400.0)
    score += min(30.0, fluff_ratio * 30.0)
    if has_low_info:
        score += 25.0
    if has_hat_boot:
        score += 25.0
    if is_parallel:
        score += 15.0
    return min(100.0, score)


def _aggregate(
    syntactic: float,
    lexicon: float,
    curvature: float | None,
    mode: EngineMode,
) -> tuple[float, EngineMode]:
    """加权聚合。返回 (风险分, 实际生效的模式)。"""
    settings = get_settings()
    if mode is EngineMode.TRI_TRACK and curvature is not None:
        risk = (
            settings.weight_tri_curvature * curvature
            + settings.weight_tri_syntactic * syntactic
            + settings.weight_tri_lexicon * lexicon
        )
        return min(100.0, max(0.0, risk)), EngineMode.TRI_TRACK

    risk = settings.weight_dual_syntactic * syntactic + settings.weight_dual_lexicon * lexicon
    return min(100.0, max(0.0, risk)), EngineMode.DUAL_TRACK


def diagnose_paragraph(
    paragraph_id: str,
    masked_text: str,
    lang: Lang | str = Lang.ZH,
    sentences: list[SentenceSpan] | None = None,
    mode: EngineMode | None = None,
) -> ParagraphDiagnosis:
    """单段三轨诊断。

    masked_text 必须是**已掩码**的文本 —— 直接传原文会让公式与引文计入句长
    并被词库扫描，分数失真。sentences 可由调用方复用已有分句结果以省一次扫描。
    """
    lang_enum = Lang(lang)
    engine_mode = mode if mode is not None else get_settings().diag_engine_mode
    spans = sentences if sentences is not None else split_sentences(masked_text, lang_enum)

    if not spans:
        return ParagraphDiagnosis(
            paragraph_id=paragraph_id,
            lang=lang_enum,
            mode=EngineMode.DUAL_TRACK,
            burstiness_cv=0.0,
            track_scores=TrackScores(syntactic=0.0, lexicon=0.0),
            overall_risk=0.0,
            risk_level=RiskLevel.LOW,
            sentences=[],
            span_issues=[],
        )

    syntactic, cv, skeletons, skeleton_source, b_issues = diagnose_track_b(spans, lang_enum)
    lexicon, c_results, c_issues = diagnose_track_c(spans, lang_enum)

    # Track A 待 M4 接入
    curvature_score: float | None = None

    overall, effective_mode = _aggregate(syntactic, lexicon, curvature_score, engine_mode)

    parallel_set = {
        i
        for i, span in enumerate(spans)
        for issue in b_issues
        if issue.issue_type is IssueType.SYNTACTIC_PARALLEL and issue.start_char == span.start_char
    }

    metrics: list[SentenceMetric] = []
    for i, span in enumerate(spans):
        c_res: TrackCSentenceResult = c_results[i]
        has_hat_boot = any(issue.issue_type is IssueType.HAT_BOOT_CLICHE for issue in c_res.issues)
        has_low_info = any(issue.issue_type is IssueType.LOW_INFO_DENSITY for issue in c_res.issues)
        clean_len = effective_length(span.text, lang_enum)
        metrics.append(
            SentenceMetric(
                sent_index=i,
                text=span.text,
                start_char=span.start_char,
                clean_char_len=clean_len,
                dep_skeleton=skeletons[i] if i < len(skeletons) else "",
                skeleton_source=skeleton_source,
                fluff_ratio=c_res.fluff_ratio,
                concrete_anchors=c_res.concrete_anchors,
                sentence_risk=_sentence_risk(
                    fluff_ratio=c_res.fluff_ratio,
                    is_parallel=i in parallel_set,
                    has_hat_boot=has_hat_boot,
                    has_low_info=has_low_info,
                    connector_hits=c_res.connector_hits,
                    clean_len=clean_len,
                ),
            )
        )

    all_issues = sorted(b_issues + c_issues, key=lambda i: (i.start_char, i.end_char))

    return ParagraphDiagnosis(
        paragraph_id=paragraph_id,
        lang=lang_enum,
        mode=effective_mode,
        burstiness_cv=cv,
        track_scores=TrackScores(syntactic=syntactic, lexicon=lexicon, curvature=curvature_score),
        overall_risk=overall,
        risk_level=RiskLevel.from_score(overall),
        sentences=metrics,
        span_issues=all_issues,
    )


def diagnose_document(
    paragraphs: list[ParsedParagraph],
    mode: EngineMode | None = None,
) -> tuple[list[ParagraphDiagnosis], DocumentRiskSummary]:
    """全篇批量诊断。

    参考文献段（section_type == "reference_skip"）直接跳过：其句长与用词分布
    与正文完全不同，纳入统计会污染全篇分数。
    """
    results: list[ParagraphDiagnosis] = []
    for para in paragraphs:
        if para.section_type == "reference_skip":
            continue
        results.append(
            diagnose_paragraph(
                paragraph_id=para.paragraph_id,
                masked_text=para.masked_text,
                lang=para.lang,
                sentences=para.sentences or None,
                mode=mode,
            )
        )

    issue_counts: dict[IssueType, int] = {}
    for diag in results:
        for issue in diag.span_issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1

    sentence_count = sum(len(d.sentences) for d in results)
    mean_risk = sum(d.overall_risk for d in results) / len(results) if results else 0.0

    summary = DocumentRiskSummary(
        paragraph_count=len(results),
        sentence_count=sentence_count,
        mean_risk=mean_risk,
        high_risk_paragraph_ids=[d.paragraph_id for d in results if d.risk_level is RiskLevel.HIGH],
        issue_counts=issue_counts,
    )
    return results, summary


def clean_char_count(masked_text: str) -> int:
    """剔除掩码后的真实字符数，用于全篇字数统计与性能基准。"""
    return len(strip_mask_tokens(masked_text).strip())
