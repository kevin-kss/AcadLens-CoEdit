"use client";

/**
 * 右栏下半：抽骨精修的两阶段流程。
 *
 * Stage 1 产出的槽位是「零幻觉」的落点：模型不知道的数据以提问形式交还作者，
 * 作者可以填也可以跳过。跳过时提示词会明确要求模型保留 [待补充] 标注，
 * 而不是用「显著提升」糊过去 —— 后者正是低信息密度的来源。
 *
 * 门禁未通过的候选也展示（置灰 + 违规原因）。静默丢弃会让用户以为系统只生成了
 * 一版，无从判断模型表现。
 */

import { useState } from "react";
import { ApiError, deconstruct, restructure } from "@/lib/api";
import { useWorkbench } from "@/lib/store";
import type { DraftCandidate, DraftStrategy, SlotKind } from "@/lib/types";

const STRATEGY_LABEL: Record<DraftStrategy, string> = {
  plain: "直叙版",
  varied: "错落版",
  blind_outline: "盲写提纲",
};

const SLOT_PLACEHOLDER: Record<SlotKind, string> = {
  numeric: "例如：120",
  dataset: "例如：CIFAR-100 长尾子集",
  metric: "例如：Top-1 准确率 76.3%",
  citation: "例如：Smith et al., 2024",
  case: "例如：某三甲医院的试点",
  condition: "例如：batch size 256，学习率 0.1",
  other: "填入真实数据",
};

function CandidateCard({ candidate }: { candidate: DraftCandidate }) {
  const accepted = !candidate.reject_reason;
  const delta = candidate.risk_after - candidate.risk_before;

  return (
    <article className={`candidate ${accepted ? "" : "candidate-rejected"}`}>
      <header>
        <h4>{STRATEGY_LABEL[candidate.strategy]}</h4>
        {accepted ? (
          <span className={`delta ${delta <= 0 ? "delta-good" : "delta-bad"}`}>
            {candidate.risk_before.toFixed(1)} → {candidate.risk_after.toFixed(1)}
            {delta <= 0 ? " ↓" : " ↑"}
          </span>
        ) : (
          <span className="delta delta-bad">未通过校验</span>
        )}
      </header>

      {candidate.masked_text ? (
        <p className="candidate-text">{candidate.masked_text}</p>
      ) : (
        <p className="panel-hint">该策略未产出草稿。</p>
      )}

      {!accepted && (
        <div className="violation-box">
          <p className="violation-head">{candidate.reject_reason}</p>
          <ul>
            {candidate.guard_result.violations.map((v, i) => (
              <li key={`${v.subject}-${i}`}>{v.message}</li>
            ))}
          </ul>
        </div>
      )}

      {candidate.retry_count > 0 && accepted && (
        <p className="panel-hint">经 {candidate.retry_count} 次修正后通过校验。</p>
      )}
    </article>
  );
}

export function AssistPanel() {
  const documentId = useWorkbench((s) => s.documentId);
  const activeId = useWorkbench((s) => s.activeParagraphId);
  const result = useWorkbench((s) => s.deconstruct);
  const drafts = useWorkbench((s) => s.restructure);
  const busy = useWorkbench((s) => s.assistBusy);
  const error = useWorkbench((s) => s.assistError);
  const setDeconstruct = useWorkbench((s) => s.setDeconstruct);
  const setRestructure = useWorkbench((s) => s.setRestructure);
  const updateSlotValue = useWorkbench((s) => s.updateSlotValue);
  const setBusy = useWorkbench((s) => s.setAssistBusy);
  const setError = useWorkbench((s) => s.setAssistError);
  const [stage, setStage] = useState<"idle" | "slots" | "drafts">("idle");

  function describe(e: unknown): string {
    if (e instanceof ApiError) {
      return e.isModelUnavailable
        ? `模型服务不可用：${e.message}。请确认 .env 中的 LLM_BASE_URL 指向可用的接口。`
        : e.message;
    }
    return e instanceof Error ? e.message : "请求失败";
  }

  async function onDeconstruct() {
    if (!documentId || !activeId) return;
    setBusy(true);
    setError(null);
    try {
      const res = await deconstruct(documentId, activeId);
      setDeconstruct(res);
      setStage("slots");
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  async function onRestructure() {
    if (!documentId || !activeId || !result) return;
    setBusy(true);
    setError(null);
    try {
      const res = await restructure(documentId, activeId, result.slots);
      setRestructure(res);
      setStage("drafts");
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  if (!activeId) {
    return (
      <section className="panel-block">
        <h2>抽骨精修</h2>
        <p className="panel-hint">选中一段后可启动协同重构。</p>
      </section>
    );
  }

  return (
    <section className="panel-block">
      <header className="panel-head">
        <h2>抽骨精修</h2>
        <p className="panel-hint">
          先抽取事实与待补数据，再按三种策略重写。改写结果须通过术语门禁才可采纳。
        </p>
      </header>

      {error && <p className="panel-error">{error}</p>}

      <div className="assist-actions">
        <button type="button" onClick={onDeconstruct} disabled={busy}>
          {busy && stage === "idle" ? "正在抽取…" : "① 抽取事实与槽位"}
        </button>
        <button
          type="button"
          onClick={onRestructure}
          disabled={busy || !result}
          title={result ? "" : "先完成第一步"}
        >
          {busy && stage === "slots" ? "正在生成…" : "② 生成候选草稿"}
        </button>
      </div>

      {result && (
        <>
          {result.facts.length > 0 && (
            <div className="fact-box">
              <h3>核心事实</h3>
              <ul>
                {result.facts.map((f, i) => (
                  <li key={i} title={f.source_quote ? `原文：${f.source_quote}` : ""}>
                    {f.subject} · {f.predicate} · {f.obj}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="slot-box">
            <h3>待补充数据（{result.slots.length}）</h3>
            {result.slots.length === 0 ? (
              <p className="panel-hint">本段没有需要补充的空洞表述。</p>
            ) : (
              <>
                <p className="panel-hint">
                  留空即跳过。跳过的项不会被编造，改写时以「[待补充]」标注。
                </p>
                {result.slots.map((slot) => (
                  <label key={slot.slot_id} className="slot-card">
                    <span className="slot-prompt">{slot.prompt}</span>
                    {slot.rationale && (
                      <span className="slot-rationale">触发原文：{slot.rationale}</span>
                    )}
                    <input
                      value={slot.value ?? ""}
                      placeholder={SLOT_PLACEHOLDER[slot.kind]}
                      onChange={(e) =>
                        updateSlotValue(slot.slot_id, e.target.value || null)
                      }
                    />
                  </label>
                ))}
              </>
            )}
          </div>
        </>
      )}

      {drafts && (
        <div className="draft-box">
          <h3>
            候选草稿（通过 {drafts.candidates.filter((c) => !c.reject_reason).length} /{" "}
            {drafts.candidates.length}）
          </h3>
          {drafts.candidates.map((c) => (
            <CandidateCard key={c.strategy} candidate={c} />
          ))}
        </div>
      )}
    </section>
  );
}
