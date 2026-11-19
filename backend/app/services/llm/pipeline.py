"""Stage 1 / Stage 2 协同重构管线。

闭环结构（技术文档 1 的时序图）：

    原段 -> Stage 1 抽骨 -> 槽位交还作者 -> Stage 2 三策略生成
         -> term_guard 硬门禁 -> 失败则带反馈重试一次
         -> 本地诊断复测风险分 -> 返回候选 + 前后对比

两条纪律：
1. **门禁不通过的草稿不丢弃**，标记 reject_reason 一并返回，前端置灰展示 ——
   静默丢弃会让用户以为系统只生成了一版，无从判断模型表现。
2. **风险分未下降不算失败**。改写可能在词库轨道上变好却在句法轨道上变差，
   这是需要用户自己判断的权衡，不是系统该代为否决的事。门禁只守事实正确性
   与术语完整性这两条硬线。
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from app.schemas.assist import (
    DeconstructResult,
    DraftCandidate,
    DraftStrategy,
    FactTriple,
    RestructureResult,
    SlotCard,
    SlotKind,
)
from app.schemas.diagnosis import Lang
from app.services.diagnostics.engine import diagnose_paragraph
from app.services.guardrails.term_guard import validate_locked_terms_preserved
from app.services.llm.client import ChatClient, LLMError
from app.services.llm.prompts import (
    build_deconstruct_messages,
    build_restructure_messages,
)

logger = logging.getLogger(__name__)

# Stage 1 用低温：抽取事实要的是忠实，不是创造力
_STAGE1_TEMPERATURE = 0.1
_STAGE2_TEMPERATURE = 0.7

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_object(raw: str) -> dict[str, object]:
    """从模型输出里取出 JSON 对象。

    即便要求了 response_format=json_object，本地小模型仍常把 JSON 包在
    ``` 围栏里或前置一句说明。这里逐级降级解析，而不是直接报错 ——
    解析失败会让整次抽骨白费一轮 Token。
    """
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 最后尝试截取第一个 { 到最后一个 } 之间的内容
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise LLMError(f"无法从模型输出解析 JSON: {raw[:200]!r}")


def _coerce_facts(payload: object) -> list[FactTriple]:
    if not isinstance(payload, list):
        return []
    facts: list[FactTriple] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("obj") or item.get("object") or "").strip()
        if not (subject or predicate or obj):
            continue
        facts.append(
            FactTriple(
                subject=subject,
                predicate=predicate,
                obj=obj,
                source_quote=str(item.get("source_quote", "")).strip(),
            )
        )
    return facts


def _coerce_slots(payload: object) -> list[SlotCard]:
    if not isinstance(payload, list):
        return []
    slots: list[SlotCard] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            continue
        raw_kind = str(item.get("kind", "other")).strip().lower()
        try:
            kind = SlotKind(raw_kind)
        except ValueError:
            kind = SlotKind.OTHER
        slots.append(
            SlotCard(
                slot_id=f"slot_{uuid.uuid4().hex[:8]}",
                kind=kind,
                prompt=prompt,
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return slots


def deconstruct_paragraph(
    paragraph_id: str,
    masked_text: str,
    client: ChatClient,
    lang: Lang | str = Lang.ZH,
    locked_terms: list[str] | None = None,
) -> DeconstructResult:
    """Stage 1：低温提取事实三元组，并为空洞表达生成填空槽位。

    masked_text 必须是已掩码文本：公式与引文以占位符形式进入提示词，
    既避免模型改写它们，也省掉大量 Token。
    """
    lang_enum = Lang(lang)
    terms = locked_terms or []
    messages = build_deconstruct_messages(masked_text, lang_enum, terms)

    raw = client.complete(
        messages,
        temperature=_STAGE1_TEMPERATURE,
        response_format_json=True,
    )
    payload = _parse_json_object(raw)
    facts = _coerce_facts(payload.get("facts"))
    slots = _coerce_slots(payload.get("slots"))

    outline = "\n".join(f"- {f.subject} | {f.predicate} | {f.obj}" for f in facts)
    return DeconstructResult(
        paragraph_id=paragraph_id,
        lang=lang_enum,
        facts=facts,
        slots=slots,
        locked_terms=terms,
        outline=outline,
    )


def _generate_one(
    strategy: DraftStrategy,
    original_masked: str,
    outline: DeconstructResult,
    client: ChatClient,
    risk_before: float,
    max_retries: int,
) -> DraftCandidate:
    """生成单份草稿，门禁不通过则带反馈重试。"""
    lang = outline.lang
    feedback = ""
    last_text = ""
    guard = validate_locked_terms_preserved(original_masked, "", outline.locked_terms)

    for attempt in range(max_retries + 1):
        messages = build_restructure_messages(
            strategy=strategy,
            outline=outline.outline,
            facts=outline.facts,
            slots=outline.slots,
            locked_terms=outline.locked_terms,
            lang=lang,
            guard_feedback=feedback,
        )
        try:
            last_text = client.complete(messages, temperature=_STAGE2_TEMPERATURE).strip()
        except LLMError as exc:
            # 单个策略失败不拖垮整次请求：标记原因、保留其余候选
            logger.warning("策略 %s 生成失败: %s", strategy.value, exc)
            return DraftCandidate(
                strategy=strategy,
                masked_text="",
                guard_result=guard,
                risk_before=risk_before,
                risk_after=risk_before,
                retry_count=attempt,
                reject_reason=f"模型调用失败：{exc}",
            )

        guard = validate_locked_terms_preserved(original_masked, last_text, outline.locked_terms)
        if guard.passed:
            risk_after = diagnose_paragraph(outline.paragraph_id, last_text, lang).overall_risk
            return DraftCandidate(
                strategy=strategy,
                masked_text=last_text,
                guard_result=guard,
                risk_before=risk_before,
                risk_after=risk_after,
                retry_count=attempt,
            )
        feedback = guard.feedback

    # 重试耗尽仍未通过：保留草稿并说明原因，由前端置灰展示
    risk_after = (
        diagnose_paragraph(outline.paragraph_id, last_text, lang).overall_risk
        if last_text
        else risk_before
    )
    return DraftCandidate(
        strategy=strategy,
        masked_text=last_text,
        guard_result=guard,
        risk_before=risk_before,
        risk_after=risk_after,
        retry_count=max_retries,
        reject_reason=f"术语门禁未通过（已重试 {max_retries} 次）",
    )


def restructure_paragraph(
    original_masked: str,
    outline: DeconstructResult,
    client: ChatClient,
    strategies: list[DraftStrategy] | None = None,
    max_retries: int = 1,
) -> RestructureResult:
    """Stage 2：按三种策略生成候选草稿，逐份过门禁并复测风险分。

    串行生成：ChatClient 是同步协议，并发需要调用方自行包装线程池 ——
    在这里起线程会让 ScriptedClient 的响应顺序不可预测，测试随之不可重现。
    """
    picked = strategies or [
        DraftStrategy.PLAIN,
        DraftStrategy.VARIED,
        DraftStrategy.BLIND_OUTLINE,
    ]
    risk_before = diagnose_paragraph(
        outline.paragraph_id, original_masked, outline.lang
    ).overall_risk

    candidates = [
        _generate_one(
            strategy=strategy,
            original_masked=original_masked,
            outline=outline,
            client=client,
            risk_before=risk_before,
            max_retries=max_retries,
        )
        for strategy in picked
    ]

    return RestructureResult(
        paragraph_id=outline.paragraph_id,
        lang=outline.lang,
        risk_before=risk_before,
        candidates=candidates,
    )
