"""Stage 1 / Stage 2 协同管线单测。

全部用 ScriptedClient，不碰网络 —— 管线的正确性与模型质量无关，混进真实
接口只会让用例变慢且不可重现。

覆盖重点：
- 零幻觉落点：空洞表达必须变成槽位；跳过的槽位必须明确告知模型不得编造
- 门禁闭环：拒绝 -> 反馈重试 -> 通过，以及重试耗尽后的降级行为
- 提示词内容：锁定术语、掩码规则、已填数据是否真的进了提示词
"""

from __future__ import annotations

import json

import pytest

from app.schemas.assist import (
    DeconstructResult,
    DraftStrategy,
    FactTriple,
    SlotCard,
    SlotKind,
)
from app.schemas.diagnosis import Lang
from app.services.llm.client import ChatMessage, LLMError, ScriptedClient
from app.services.llm.pipeline import (
    _parse_json_object,
    deconstruct_paragraph,
    restructure_paragraph,
)
from app.services.llm.prompts import (
    build_deconstruct_messages,
    build_restructure_messages,
)

ORIGINAL = (
    "在当今社会，多头自注意力机制的研究日益重要。值得注意的是，该方法具有多维度的应用价值。"
    "此外，其显著提升了系统的整体效能 {{CITE_1}}。综上所述，本研究具有重要的理论意义。"
)
LOCKED = ["多头自注意力机制"]

STAGE1_JSON = json.dumps(
    {
        "facts": [
            {
                "subject": "多头自注意力机制",
                "predicate": "提升",
                "obj": "系统效能",
                "source_quote": "其显著提升了系统的整体效能",
            }
        ],
        "slots": [
            {
                "kind": "metric",
                "prompt": "请补充「提升效能」的具体指标与数值",
                "rationale": "显著提升了系统的整体效能",
            },
            {
                "kind": "dataset",
                "prompt": "请补充实验所用数据集",
                "rationale": "多维度的应用价值",
            },
        ],
    },
    ensure_ascii=False,
)

GOOD_DRAFT = "多头自注意力机制把 Top-1 准确率从 71.8% 提升到 76.3% {{CITE_1}}。"
BAD_DRAFT = "该注意力方法把准确率提到了 76.3%。"


def make_outline(**overrides: object) -> DeconstructResult:
    defaults: dict[str, object] = {
        "paragraph_id": "p1",
        "lang": Lang.ZH,
        "facts": [FactTriple(subject="多头自注意力机制", predicate="提升", obj="效能")],
        "slots": [],
        "locked_terms": list(LOCKED),
        "outline": "- 多头自注意力机制 | 提升 | 效能",
    }
    defaults.update(overrides)
    return DeconstructResult(**defaults)


class TestJsonParsing:
    """本地小模型常把 JSON 包在围栏里或前置说明，解析必须逐级降级。"""

    def test_plain_json(self):
        assert _parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_without_language(self):
        assert _parse_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_prefixed_prose(self):
        assert _parse_json_object('好的，结果如下：\n{"a": 1}') == {"a": 1}

    def test_trailing_prose(self):
        assert _parse_json_object('{"a": 1}\n以上就是结果。') == {"a": 1}

    def test_unparseable_raises(self):
        with pytest.raises(LLMError, match="无法从模型输出解析 JSON"):
            _parse_json_object("这段文本里完全没有 JSON")

    def test_json_array_rejected(self):
        """顶层必须是对象：数组说明模型没按 schema 输出。"""
        with pytest.raises(LLMError):
            _parse_json_object("[1, 2, 3]")


class TestStage1Deconstruct:
    def test_extracts_facts_and_slots(self):
        client = ScriptedClient(responses=[STAGE1_JSON])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, LOCKED)
        assert len(result.facts) == 1
        assert result.facts[0].subject == "多头自注意力机制"
        assert len(result.slots) == 2
        assert {s.kind for s in result.slots} == {SlotKind.METRIC, SlotKind.DATASET}

    def test_slots_start_unfilled(self):
        client = ScriptedClient(responses=[STAGE1_JSON])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, LOCKED)
        assert all(not s.is_filled for s in result.slots)

    def test_slot_ids_are_unique(self):
        client = ScriptedClient(responses=[STAGE1_JSON])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, LOCKED)
        assert len({s.slot_id for s in result.slots}) == len(result.slots)

    def test_outline_built_from_facts(self):
        client = ScriptedClient(responses=[STAGE1_JSON])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, LOCKED)
        assert "多头自注意力机制" in result.outline

    def test_locked_terms_carried_forward(self):
        client = ScriptedClient(responses=[STAGE1_JSON])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, LOCKED)
        assert result.locked_terms == LOCKED

    def test_unknown_slot_kind_falls_back_to_other(self):
        payload = json.dumps(
            {"facts": [], "slots": [{"kind": "wat", "prompt": "补充什么"}]},
            ensure_ascii=False,
        )
        client = ScriptedClient(responses=[payload])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, [])
        assert result.slots[0].kind is SlotKind.OTHER

    def test_object_key_alias_accepted(self):
        """模型常输出 "object" 而非 "obj"。"""
        payload = json.dumps(
            {"facts": [{"subject": "A", "predicate": "B", "object": "C"}], "slots": []},
            ensure_ascii=False,
        )
        client = ScriptedClient(responses=[payload])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, [])
        assert result.facts[0].obj == "C"

    def test_malformed_entries_skipped_not_fatal(self):
        payload = json.dumps(
            {
                "facts": ["不是对象", {"subject": "", "predicate": "", "obj": ""}],
                "slots": ["不是对象", {"prompt": ""}, {"prompt": "有效槽位"}],
            },
            ensure_ascii=False,
        )
        client = ScriptedClient(responses=[payload])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, [])
        assert result.facts == []
        assert len(result.slots) == 1

    def test_missing_keys_tolerated(self):
        client = ScriptedClient(responses=["{}"])
        result = deconstruct_paragraph("p1", ORIGINAL, client, Lang.ZH, [])
        assert result.facts == []
        assert result.slots == []

    def test_uses_low_temperature(self):
        """抽事实要的是忠实而非创造力。"""
        recorded: dict[str, float] = {}

        class Recorder:
            def complete(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.2,
                max_tokens: int | None = None,
                response_format_json: bool = False,
            ) -> str:
                recorded["temperature"] = temperature
                return STAGE1_JSON

        deconstruct_paragraph("p1", ORIGINAL, Recorder(), Lang.ZH, LOCKED)
        assert recorded["temperature"] <= 0.2


class TestStage1Prompt:
    def test_prompt_forbids_fabrication(self):
        messages = build_deconstruct_messages(ORIGINAL, Lang.ZH, LOCKED)
        combined = "\n".join(m.content for m in messages)
        assert "严禁捏造" in combined

    def test_prompt_protects_mask_tokens(self):
        messages = build_deconstruct_messages(ORIGINAL, Lang.ZH, LOCKED)
        combined = "\n".join(m.content for m in messages)
        assert "不可变占位符" in combined
        assert "{{CITE_1}}" in combined

    def test_prompt_lists_locked_terms(self):
        messages = build_deconstruct_messages(ORIGINAL, Lang.ZH, LOCKED)
        combined = "\n".join(m.content for m in messages)
        assert "多头自注意力机制" in combined

    def test_prompt_demands_slots_over_guessing(self):
        messages = build_deconstruct_messages(ORIGINAL, Lang.ZH, [])
        combined = "\n".join(m.content for m in messages)
        assert "槽位" in combined

    def test_english_prompt(self):
        messages = build_deconstruct_messages("Some text.", Lang.EN, ["RMSE"])
        combined = "\n".join(m.content for m in messages)
        assert "Never fabricate" in combined
        assert "RMSE" in combined


class TestStage2Prompt:
    def test_filled_slot_values_included(self):
        slots = [
            SlotCard(slot_id="s1", kind=SlotKind.METRIC, prompt="指标", value="76.3%"),
        ]
        messages = build_restructure_messages(DraftStrategy.PLAIN, "", [], slots, LOCKED, Lang.ZH)
        combined = "\n".join(m.content for m in messages)
        assert "76.3%" in combined
        assert "必须原样使用" in combined

    def test_skipped_slot_forbids_fabrication(self):
        """缺数据时不得编造数值 —— 零捏造的底线。"""
        slots = [SlotCard(slot_id="s1", kind=SlotKind.METRIC, prompt="指标", value=None)]
        messages = build_restructure_messages(DraftStrategy.PLAIN, "", [], slots, LOCKED, Lang.ZH)
        combined = "\n".join(m.content for m in messages)
        assert "不得编造任何数值" in combined

    def test_skipped_slot_keeps_the_claim(self):
        """论点结构是作者的，缺证据不等于可以删句子。"""
        slots = [SlotCard(slot_id="s1", kind=SlotKind.METRIC, prompt="指标", value=None)]
        messages = build_restructure_messages(DraftStrategy.PLAIN, "", [], slots, LOCKED, Lang.ZH)
        combined = "\n".join(m.content for m in messages)
        assert "保留该论点" in combined
        assert "不得改变原意" in combined

    def test_skipped_slot_drops_degree_modifiers(self):
        """去掉无证据的程度词，这是既不编造、不堆套话、也不改原意的唯一出路。

        不能改用「显著」「多维度」填空：这批词正在本项目的空泛词表与帽句规则里，
        写回去会被 Stage 2 后的诊断复测扣分，改写器与诊断器自相矛盾。
        """
        slots = [SlotCard(slot_id="s1", kind=SlotKind.METRIC, prompt="指标", value=None)]
        messages = build_restructure_messages(DraftStrategy.PLAIN, "", [], slots, LOCKED, Lang.ZH)
        combined = "\n".join(m.content for m in messages)
        assert "去掉没有证据支撑的程度修饰词" in combined
        for modifier in ("显著", "多维度", "广阔"):
            assert modifier in combined, f"程度词清单缺 {modifier}"

    def test_skipped_slot_forbids_placeholder_markers(self):
        """不写 [待补充]：原文常本就没有那些数据，作者也不打算补。

        满屏占位标注等于要求作者补一份根本不存在的数据。
        """
        slots = [SlotCard(slot_id="s1", kind=SlotKind.METRIC, prompt="指标", value=None)]
        messages = build_restructure_messages(DraftStrategy.PLAIN, "", [], slots, LOCKED, Lang.ZH)
        combined = "\n".join(m.content for m in messages)
        assert "不得写" in combined and "待补充" in combined

    def test_short_sentence_must_be_complete(self):
        """短句约束必须要求语义完整。

        早期提示词只规定「10 字以内」，模型为凑字数产出过「价值具体。」这种病句。
        """
        messages = build_restructure_messages(DraftStrategy.PLAIN, "", [], [], [], Lang.ZH)
        rule = messages[0].content
        assert "语义完整" in rule
        assert "凑不出" in rule
        assert "10 字以内的短判断句" not in rule

    def test_prompt_requires_breaking_uniform_length(self):
        """打破句长均匀是降 AI 味的主要手段，收敛为一版时不能随错落版一起删掉。"""
        rule = build_restructure_messages(DraftStrategy.PLAIN, "", [], [], [], Lang.ZH)[0].content
        assert "打破句长" in rule
        assert "主干结构不得雷同" in rule

    def test_prompt_forbids_expansion(self):
        """错落版曾为调节奏把信息摊长，实质是扩写。唯一版须明确禁止。"""
        rule = build_restructure_messages(DraftStrategy.PLAIN, "", [], [], [], Lang.ZH)[0].content
        assert "不要扩写" in rule
        assert "不超过原文" in rule

    def test_guard_feedback_included_on_retry(self):
        messages = build_restructure_messages(
            DraftStrategy.PLAIN, "", [], [], LOCKED, Lang.ZH, guard_feedback="- 丢了术语"
        )
        combined = "\n".join(m.content for m in messages)
        assert "丢了术语" in combined
        assert "被自动校验拒绝" in combined

    @pytest.mark.parametrize("strategy", list(DraftStrategy))
    def test_each_strategy_has_distinct_rule(self, strategy: DraftStrategy):
        messages = build_restructure_messages(strategy, "", [], [], [], Lang.ZH)
        assert messages[0].content

    def test_only_one_strategy_exists(self):
        """只保留一版：用户要能直接复制使用的成稿，并列多版是把判断推给用户。

        实测中错落版风险分（36.2）反而高于直叙版（22.5），排序都反了。
        """
        assert list(DraftStrategy) == [DraftStrategy.PLAIN]

    def test_prompts_forbid_explicit_connectives(self):
        messages = build_restructure_messages(DraftStrategy.PLAIN, "", [], [], [], Lang.ZH)
        assert "显性连接词" in messages[0].content


class TestStage2GuardLoop:
    def test_passing_draft_accepted_without_retry(self):
        client = ScriptedClient(responses=[GOOD_DRAFT])
        result = restructure_paragraph(ORIGINAL, make_outline(), client, [DraftStrategy.PLAIN])
        candidate = result.candidates[0]
        assert candidate.accepted
        assert candidate.retry_count == 0
        assert candidate.guard_result.passed

    def test_rejected_draft_retried_then_accepted(self):
        client = ScriptedClient(responses=[BAD_DRAFT, GOOD_DRAFT])
        result = restructure_paragraph(
            ORIGINAL, make_outline(), client, [DraftStrategy.PLAIN], max_retries=1
        )
        candidate = result.candidates[0]
        assert candidate.accepted
        assert candidate.retry_count == 1
        assert len(client.calls) == 2

    def test_retry_prompt_carries_feedback(self):
        client = ScriptedClient(responses=[BAD_DRAFT, GOOD_DRAFT])
        restructure_paragraph(
            ORIGINAL, make_outline(), client, [DraftStrategy.PLAIN], max_retries=1
        )
        retry_prompt = "\n".join(m.content for m in client.calls[1])
        assert "被自动校验拒绝" in retry_prompt
        assert "多头自注意力机制" in retry_prompt

    def test_exhausted_retries_marked_rejected_but_kept(self):
        """不丢弃：静默丢弃会让用户以为系统只生成了一版。"""
        client = ScriptedClient(default_response=BAD_DRAFT)
        result = restructure_paragraph(
            ORIGINAL, make_outline(), client, [DraftStrategy.PLAIN], max_retries=1
        )
        candidate = result.candidates[0]
        assert not candidate.accepted
        assert candidate.reject_reason
        assert candidate.masked_text == BAD_DRAFT
        assert candidate.guard_result.violations

    def test_llm_failure_is_marked_not_raised(self):
        """模型调用失败应标记在候选上，而不是向上抛错中断整个请求。"""

        class FlakyClient:
            def __init__(self) -> None:
                self.count = 0

            def complete(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.2,
                max_tokens: int | None = None,
                response_format_json: bool = False,
            ) -> str:
                self.count += 1
                if self.count == 1:
                    raise LLMError("模拟接口超时")
                return GOOD_DRAFT

        result = restructure_paragraph(ORIGINAL, make_outline(), FlakyClient(), max_retries=0)
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert not candidate.accepted
        assert "模型调用失败" in candidate.reject_reason
        # 仍返回结果对象而非抛错，前端据此提示而不是白屏
        assert result.risk_before > 0

    def test_no_locked_terms_still_checks_masks(self):
        outline = make_outline(locked_terms=[])
        client = ScriptedClient(default_response="改写后的文本，丢了引文。")
        result = restructure_paragraph(
            ORIGINAL, outline, client, [DraftStrategy.PLAIN], max_retries=0
        )
        assert not result.candidates[0].accepted


class TestStage2RiskComparison:
    def test_risk_before_and_after_recorded(self):
        client = ScriptedClient(responses=[GOOD_DRAFT])
        result = restructure_paragraph(ORIGINAL, make_outline(), client, [DraftStrategy.PLAIN])
        candidate = result.candidates[0]
        assert candidate.risk_before == result.risk_before
        assert candidate.risk_before > 0
        assert candidate.risk_delta == candidate.risk_after - candidate.risk_before

    def test_cleaner_draft_lowers_risk(self):
        client = ScriptedClient(responses=[GOOD_DRAFT])
        result = restructure_paragraph(ORIGINAL, make_outline(), client, [DraftStrategy.PLAIN])
        assert result.candidates[0].risk_after < result.candidates[0].risk_before

    def test_risk_not_dropping_is_not_a_rejection(self):
        """风险分未降是用户要权衡的事，不是系统该代为否决的。

        门禁只守事实正确性与术语完整性两条硬线。
        """
        wordy = (
            "多头自注意力机制。此外，值得注意的是，综上所述，由此可见 {{CITE_1}}。"
            "总而言之，显而易见，不言而喻。"
        )
        client = ScriptedClient(responses=[wordy])
        result = restructure_paragraph(ORIGINAL, make_outline(), client, [DraftStrategy.PLAIN])
        candidate = result.candidates[0]
        assert candidate.guard_result.passed
        assert candidate.accepted

    def test_single_candidate_by_default(self):
        """默认只生成一版，不再并列三版。"""
        client = ScriptedClient(default_response=GOOD_DRAFT)
        result = restructure_paragraph(ORIGINAL, make_outline(), client)
        assert [c.strategy for c in result.candidates] == [DraftStrategy.PLAIN]
        assert len(result.accepted_candidates) == 1

    def test_only_one_model_call_per_draft(self):
        """一版一次调用：三版并列时要调三次，Token 成本是三倍。"""
        client = ScriptedClient(default_response=GOOD_DRAFT)
        restructure_paragraph(ORIGINAL, make_outline(), client)
        assert len(client.calls) == 1


class TestScriptedClient:
    def test_records_calls(self):
        client = ScriptedClient(responses=["a", "b"])
        client.complete([ChatMessage("user", "one")])
        client.complete([ChatMessage("user", "two")])
        assert len(client.calls) == 2
        assert client.last_prompt == "two"

    def test_raises_when_exhausted(self):
        client = ScriptedClient(responses=["only"])
        client.complete([ChatMessage("user", "x")])
        with pytest.raises(LLMError):
            client.complete([ChatMessage("user", "y")])

    def test_default_response_reused(self):
        client = ScriptedClient(default_response="same")
        assert client.complete([]) == "same"
        assert client.complete([]) == "same"
