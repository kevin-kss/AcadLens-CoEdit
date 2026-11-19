"use client";

/**
 * 左栏：候选术语与锁定表。
 *
 * C-Value 分数、频次与嵌套次数都展示出来，这是白盒原则的一部分 ——
 * 用户要能看懂「为什么系统推荐这个词」，而不是面对一串没有依据的建议。
 */

import { useState } from "react";
import { setLockedTerms as pushTerms } from "@/lib/api";
import { useWorkbench } from "@/lib/store";

export function TermPanel() {
  const documentId = useWorkbench((s) => s.documentId);
  const suggested = useWorkbench((s) => s.suggestedTerms);
  const locked = useWorkbench((s) => s.lockedTerms);
  const toggleTerm = useWorkbench((s) => s.toggleTerm);
  const setTerms = useWorkbench((s) => s.setLockedTerms);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function persist(next: string[]) {
    if (!documentId) return;
    setSaving(true);
    setError(null);
    try {
      await pushTerms(documentId, next);
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  function onToggle(term: string) {
    const next = locked.includes(term)
      ? locked.filter((t) => t !== term)
      : [...locked, term];
    toggleTerm(term);
    void persist(next);
  }

  function onAdd() {
    const term = draft.trim();
    if (!term || locked.includes(term)) {
      setDraft("");
      return;
    }
    const next = [...locked, term];
    setTerms(next);
    setDraft("");
    void persist(next);
  }

  return (
    <aside className="panel panel-left">
      <header className="panel-head">
        <h2>锁定术语</h2>
        <p className="panel-hint">
          锁定的术语在改写时必须原样保留，由代码级校验强制执行，不依赖模型自觉。
        </p>
      </header>

      <div className="term-add">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onAdd();
          }}
          placeholder="手动添加术语"
          aria-label="手动添加术语"
        />
        <button type="button" onClick={onAdd} disabled={!draft.trim()}>
          添加
        </button>
      </div>

      {error && <p className="panel-error">{error}</p>}
      {saving && <p className="panel-hint">正在保存…</p>}

      {locked.length > 0 && (
        <section className="term-section">
          <h3>已锁定 ({locked.length})</h3>
          <ul className="term-list">
            {locked.map((term) => (
              <li key={term}>
                <label>
                  <input type="checkbox" checked onChange={() => onToggle(term)} />
                  <span className="term-name">{term}</span>
                </label>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="term-section">
        <h3>候选术语</h3>
        {suggested.length === 0 ? (
          <p className="panel-hint">
            暂无候选。术语需在全文出现两次以上才会被提取。
          </p>
        ) : (
          <ul className="term-list">
            {suggested.map((candidate) => (
              <li key={candidate.term}>
                <label>
                  <input
                    type="checkbox"
                    checked={locked.includes(candidate.term)}
                    onChange={() => onToggle(candidate.term)}
                  />
                  <span className="term-name">{candidate.term}</span>
                </label>
                <span
                  className="term-meta"
                  title={`C-Value ${candidate.score.toFixed(2)}：出现 ${candidate.frequency} 次，${candidate.word_count} 词${candidate.nested_in > 0 ? `，作为更长术语的一部分出现 ${candidate.nested_in} 次（已折减）` : ""}`}
                >
                  {candidate.score.toFixed(1)} · {candidate.frequency}次
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
