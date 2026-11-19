"use client";

/**
 * 右栏上半：当前段的风险分解。
 *
 * 白盒的落点：不只给总分，而是把各轨道子分、句长变异系数、骨架来源与逐句
 * 风险都摊开。用户要能回答「这个分数是怎么来的」，否则诊断就只是另一个
 * 不可解释的黑盒判决。
 */

import { useWorkbench } from "@/lib/store";
import type { IssueType, ParagraphView, RiskLevel } from "@/lib/types";

const RISK_LABEL: Record<RiskLevel, string> = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
};

const ISSUE_LABEL: Record<IssueType, string> = {
  ai_connector: "套话连接词",
  hat_boot_cliche: "帽句/升华句",
  low_info_density: "信息密度低",
  uniform_length: "句长过于均匀",
  syntactic_parallel: "句式雷同",
  high_curvature: "概率曲率异常",
};

function RiskBar({ label, value, hint }: { label: string; value: number; hint: string }) {
  return (
    <div className="metric" title={hint}>
      <div className="metric-head">
        <span>{label}</span>
        <span className="metric-value">{value.toFixed(1)}</span>
      </div>
      <div className="metric-track">
        <div className="metric-fill" style={{ width: `${Math.min(100, value)}%` }} />
      </div>
    </div>
  );
}

export function RiskInspector() {
  const paragraphs = useWorkbench((s) => s.paragraphs);
  const activeId = useWorkbench((s) => s.activeParagraphId);
  const summary = useWorkbench((s) => s.summary);
  const recomputing = useWorkbench((s) => s.recomputingIds);

  const active: ParagraphView | undefined = paragraphs.find(
    (p) => p.paragraph_id === activeId,
  );

  if (!active) {
    return (
      <section className="panel-block">
        <h2>风险分解</h2>
        <p className="panel-hint">把光标放进某一段，这里会显示它的诊断依据。</p>
      </section>
    );
  }

  const d = active.diagnosis;
  const busy = recomputing.has(active.paragraph_id);
  const source = d.sentences[0]?.skeleton_source ?? "none";

  const issueCounts = new Map<IssueType, number>();
  for (const issue of d.span_issues) {
    issueCounts.set(issue.issue_type, (issueCounts.get(issue.issue_type) ?? 0) + 1);
  }

  return (
    <section className="panel-block">
      <header className="panel-head">
        <h2>
          风险分解
          {busy && <span className="spinner" aria-label="正在重算" />}
        </h2>
        {summary && (
          <p className="panel-hint">
            全篇均值 {summary.mean_risk.toFixed(1)} · 高风险段 {summary.high_risk_paragraph_ids.length} 个
          </p>
        )}
      </header>

      <div className={`risk-headline risk-${d.risk_level}`}>
        <span className="risk-score">{d.overall_risk.toFixed(1)}</span>
        <span className="risk-label">{RISK_LABEL[d.risk_level]}</span>
      </div>

      <RiskBar
        label="句法轨道"
        value={d.track_scores.syntactic}
        hint="句长变异系数与依存骨架同构度。分数高说明句子长度均匀、句式雷同。"
      />
      <RiskBar
        label="词库轨道"
        value={d.track_scores.lexicon}
        hint="套话连接词密度、帽句命中与信息空洞度。"
      />
      {d.track_scores.curvature !== null && (
        <RiskBar
          label="概率曲率"
          value={d.track_scores.curvature}
          hint="本地小模型的概率曲率（Track A）。"
        />
      )}

      <dl className="fact-grid">
        <dt>句长变异系数</dt>
        <dd title="人类学术写作通常大于 0.30；低于 0.18 说明句长过于均匀。">
          {d.burstiness_cv.toFixed(3)}
        </dd>
        <dt>骨架来源</dt>
        <dd
          title={
            source === "dep"
              ? "spaCy 真实依存分析。"
              : "未安装 spaCy 模型，降级为词性近似骨架；阈值与真依存不同，分数可比性较弱。"
          }
        >
          {source === "dep" ? "真实依存" : source === "pos" ? "词性近似（降级）" : "无"}
        </dd>
        <dt>运行模式</dt>
        <dd>{d.mode === "tri_track" ? "三轨" : "双轨"}</dd>
      </dl>

      {issueCounts.size > 0 && (
        <ul className="issue-legend">
          {[...issueCounts.entries()].map(([type, count]) => (
            <li key={type}>
              <span className={`legend-swatch issue-${type.replace(/_/g, "-")}`} />
              {ISSUE_LABEL[type]} × {count}
            </li>
          ))}
        </ul>
      )}

      <details className="sentence-detail">
        <summary>逐句风险（{d.sentences.length} 句）</summary>
        <ol className="sentence-list">
          {d.sentences.map((s) => (
            <li key={s.sent_index}>
              <div className="sentence-row">
                <span className="sentence-risk" data-level={s.sentence_risk > 50 ? "high" : s.sentence_risk > 25 ? "mid" : "low"}>
                  {s.sentence_risk.toFixed(0)}
                </span>
                <span className="sentence-text">{s.text}</span>
              </div>
              <span className="sentence-meta">
                {s.clean_char_len} 字 · 空洞度 {s.fluff_ratio.toFixed(2)} · 具象锚点 {s.concrete_anchors}
              </span>
            </li>
          ))}
        </ol>
      </details>
    </section>
  );
}
