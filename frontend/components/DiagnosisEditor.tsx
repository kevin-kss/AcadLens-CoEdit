"use client";

/**
 * TipTap 编辑器 + 风险装饰层。
 *
 * 实时重算走 800ms 防抖：后端单段诊断实测 2.3ms，但每次按键都发请求会让网络
 * 往返成为瓶颈，且用户打字中途的半截句子诊断出来没有意义。
 *
 * 只重算「光标所在的那一段」，不重跑全文 —— 这是满足响应预算的前提。
 */

import { useEffect, useRef } from "react";
import { EditorContent, useEditor } from "@tiptap/react";
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import History from "@tiptap/extension-history";
import { buildDecorations, collectParagraphPositions } from "@/lib/decorations";
import { useWorkbench } from "@/lib/store";
import { updateParagraph } from "@/lib/api";
import type { ParagraphView } from "@/lib/types";

const DEBOUNCE_MS = 800;

const decorationKey = new PluginKey("risk-decorations");

/**
 * 装饰层插件。
 *
 * 诊断数据通过闭包读 store 而非 plugin state：装饰只依赖最新的 paragraphs，
 * 每次 store 更新后由外部调用 forceUpdate 触发重绘，不需要在插件里维护副本。
 */
function createDecorationExtension(getParagraphs: () => ParagraphView[]) {
  return Extension.create({
    name: "riskDecorations",
    addProseMirrorPlugins() {
      return [
        new Plugin({
          key: decorationKey,
          props: {
            decorations(state) {
              return buildDecorations(state.doc, getParagraphs());
            },
          },
        }),
      ];
    },
  });
}

export function DiagnosisEditor() {
  const documentId = useWorkbench((s) => s.documentId);
  const paragraphs = useWorkbench((s) => s.paragraphs);
  const lang = useWorkbench((s) => s.lang);
  const activeParagraphId = useWorkbench((s) => s.activeParagraphId);
  const setActiveParagraph = useWorkbench((s) => s.setActiveParagraph);
  const replaceParagraph = useWorkbench((s) => s.replaceParagraph);
  const markRecomputing = useWorkbench((s) => s.markRecomputing);

  // 用 ref 读最新值：ProseMirror 插件与防抖回调都在闭包里，直接捕获 state
  // 会拿到创建时的旧快照
  const paragraphsRef = useRef(paragraphs);
  paragraphsRef.current = paragraphs;
  const documentIdRef = useRef(documentId);
  documentIdRef.current = documentId;

  const timersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>());

  const editor = useEditor({
    // 不用 StarterKit：它带了标题、列表、引用等节点，而工作台的文档模型刻意
    // 限制为「只有段落」—— 后端的 paragraphs 数组与编辑器节点必须一一对应，
    // 允许用户插入列表会立刻打破这个前提。
    extensions: [
      Document,
      Paragraph,
      Text,
      History,
      createDecorationExtension(() => paragraphsRef.current),
    ],
    content: { type: "doc", content: [] },
    editorProps: {
      attributes: { class: "editor-surface", spellcheck: "false" },
    },
    // Next.js SSR 下必须关掉，否则 hydration 不匹配
    immediatelyRender: false,
    onSelectionUpdate({ editor: ed }) {
      const positions = collectParagraphPositions(
        ed.state.doc,
        paragraphsRef.current.map((p) => p.paragraph_id),
      );
      const cursor = ed.state.selection.from;
      // 找到光标落在哪个段落节点内
      const hit = positions.find(
        ({ pos, node }) => cursor >= pos && cursor <= pos + node.nodeSize,
      );
      if (hit && hit.paragraphId !== useWorkbench.getState().activeParagraphId) {
        setActiveParagraph(hit.paragraphId);
      }
    },
    onUpdate({ editor: ed, transaction }) {
      if (!transaction.docChanged) return;
      const docId = documentIdRef.current;
      if (!docId) return;

      const positions = collectParagraphPositions(
        ed.state.doc,
        paragraphsRef.current.map((p) => p.paragraph_id),
      );
      const cursor = ed.state.selection.from;
      const hit = positions.find(
        ({ pos, node }) => cursor >= pos && cursor <= pos + node.nodeSize,
      );
      if (!hit) return;

      const { paragraphId, node } = hit;
      const text = node.textContent;

      const timers = timersRef.current;
      const existing = timers.get(paragraphId);
      if (existing) clearTimeout(existing);

      timers.set(
        paragraphId,
        setTimeout(() => {
          timers.delete(paragraphId);
          markRecomputing(paragraphId, true);
          updateParagraph(docId, paragraphId, text)
            .then((view) => {
              replaceParagraph(view);
              // 诊断回来后强制重绘装饰层
              ed.view.dispatch(ed.state.tr.setMeta(decorationKey, Date.now()));
            })
            .catch(() => {
              // 重算失败不打断编辑：装饰层会因文本不匹配而暂时不显示，
              // 下次编辑会再试
            })
            .finally(() => markRecomputing(paragraphId, false));
        }, DEBOUNCE_MS),
      );
    },
  });

  // 文档载入后把段落写进编辑器
  const loadedDocRef = useRef<string | null>(null);
  useEffect(() => {
    if (!editor || !documentId) return;
    if (loadedDocRef.current === documentId) return;
    loadedDocRef.current = documentId;

    editor.commands.setContent({
      type: "doc",
      content: paragraphs.map((p) => ({
        type: "paragraph",
        content: p.masked_text ? [{ type: "text", text: p.masked_text }] : [],
      })),
    });
  }, [editor, documentId, paragraphs]);

  // store 里的诊断更新后重绘装饰层（例如锁定术语变更导致全篇重诊断）
  useEffect(() => {
    if (!editor) return;
    editor.view.dispatch(editor.state.tr.setMeta(decorationKey, Date.now()));
  }, [editor, paragraphs]);

  // 卸载时清掉未触发的防抖定时器
  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((t) => clearTimeout(t));
      timers.clear();
    };
  }, []);

  useEffect(() => {
    if (editor) editor.setOptions({ editorProps: { attributes: { lang } } });
  }, [editor, lang]);

  if (!documentId) {
    return (
      <div className="editor-empty">
        <p>粘贴或导入稿件后，这里会显示逐句诊断。</p>
      </div>
    );
  }

  return (
    <div className="editor-wrap" data-active={activeParagraphId ?? ""}>
      <EditorContent editor={editor} />
    </div>
  );
}
