"use client";

/**
 * 顶栏：导入稿件、语种选择与导出。
 *
 * 只收纯文本：.docx 的样式保真解析在后端尚未落地（M4），这里不做一个
 * 能选文件但丢样式的上传框 —— 那会让用户以为样式已保留。
 */

import { useState } from "react";
import { parseDocument } from "@/lib/api";
import { useWorkbench } from "@/lib/store";
import type { Lang } from "@/lib/types";

const SAMPLE = `在当今社会，随着人工智能技术的不断发展，多头自注意力机制的研究日益重要。值得注意的是，该方法具有多维度的应用价值。此外，其显著提升了系统的整体效能 [12]。综上所述，本研究具有重要的理论意义与广阔的应用前景。

我们在 ImageNet-1k 上训练 ResNet-50，批大小 256，学习率 0.1，共 90 个 epoch。Top-1 准确率为 76.3%。去掉残差连接后该数值降至 71.8%，且训练在第 40 个 epoch 后不再收敛，其梯度范数见 $\\|\\nabla L\\|_2$。`;

export function ImportBar() {
  const documentId = useWorkbench((s) => s.documentId);
  const loadDocument = useWorkbench((s) => s.loadDocument);
  const reset = useWorkbench((s) => s.reset);
  const [text, setText] = useState("");
  const [lang, setLang] = useState<Lang>("zh");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(true);

  async function onImport(payload: string) {
    if (!payload.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const doc = await parseDocument(payload, lang);
      loadDocument(doc);
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "导入失败");
    } finally {
      setBusy(false);
    }
  }

  function onExport() {
    if (!documentId) return;
    const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
    window.open(`${base}/api/v1/documents/${documentId}/export`, "_blank");
  }

  return (
    <header className="topbar">
      <div className="topbar-row">
        <h1>
          AcadLens<span className="topbar-accent">CoEdit</span>
        </h1>
        <div className="topbar-actions">
          <select
            value={lang}
            onChange={(e) => setLang(e.target.value as Lang)}
            aria-label="稿件语种"
            disabled={busy}
          >
            <option value="zh">中文</option>
            <option value="en">English</option>
          </select>
          {documentId && (
            <>
              <button type="button" onClick={onExport}>
                导出（还原公式与引文）
              </button>
              <button
                type="button"
                onClick={() => {
                  reset();
                  setOpen(true);
                  setText("");
                }}
              >
                换一篇
              </button>
            </>
          )}
          {!open && !documentId && (
            <button type="button" onClick={() => setOpen(true)}>
              导入稿件
            </button>
          )}
        </div>
      </div>

      {open && (
        <div className="import-area">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="粘贴稿件正文，按空行分段。公式与引文会被自动隔离保护。"
            rows={6}
            disabled={busy}
          />
          <div className="import-actions">
            <button
              type="button"
              onClick={() => onImport(text)}
              disabled={busy || !text.trim()}
            >
              {busy ? "正在诊断…" : "开始诊断"}
            </button>
            <button
              type="button"
              className="btn-ghost"
              onClick={() => {
                setText(SAMPLE);
                void onImport(SAMPLE);
              }}
              disabled={busy}
            >
              用示例稿件试一下
            </button>
            <span className="panel-hint">
              .docx 导入待样式保真解析落地后接入，当前仅支持纯文本。
            </span>
          </div>
          {error && <p className="panel-error">{error}</p>}
        </div>
      )}
    </header>
  );
}
