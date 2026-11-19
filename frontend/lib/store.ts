/**
 * 工作台的全局状态（Zustand）。
 *
 * 一个刻意的取舍：段落的**诊断结果**放在 store 里，而段落的**文本**由 TipTap
 * 自己持有。两处都存文本会立刻产生同步问题（谁是真相？），而 TipTap 的
 * ProseMirror 文档本身就是编辑状态的唯一真相。store 只保存「后端算出来的东西」。
 */

import { create } from "zustand";
import type {
  DeconstructResult,
  DocumentResponse,
  DocumentSummary,
  Lang,
  ParagraphView,
  RestructureResult,
  SlotCard,
  TermCandidate,
} from "./types";

interface WorkbenchState {
  documentId: string | null;
  lang: Lang;
  paragraphs: ParagraphView[];
  suggestedTerms: TermCandidate[];
  lockedTerms: string[];
  summary: DocumentSummary | null;

  activeParagraphId: string | null;
  /** 正在重算的段落。用于只在该段显示 spinner，而非整页遮罩。 */
  recomputingIds: Set<string>;

  deconstruct: DeconstructResult | null;
  restructure: RestructureResult | null;
  assistBusy: boolean;
  assistError: string | null;

  loadDocument: (doc: DocumentResponse) => void;
  setActiveParagraph: (id: string | null) => void;
  replaceParagraph: (view: ParagraphView) => void;
  markRecomputing: (id: string, busy: boolean) => void;
  toggleTerm: (term: string) => void;
  setLockedTerms: (terms: string[]) => void;
  setDeconstruct: (result: DeconstructResult | null) => void;
  updateSlotValue: (slotId: string, value: string | null) => void;
  setRestructure: (result: RestructureResult | null) => void;
  setAssistBusy: (busy: boolean) => void;
  setAssistError: (message: string | null) => void;
  reset: () => void;
}

const initial = {
  documentId: null,
  lang: "zh" as Lang,
  paragraphs: [],
  suggestedTerms: [],
  lockedTerms: [],
  summary: null,
  activeParagraphId: null,
  recomputingIds: new Set<string>(),
  deconstruct: null,
  restructure: null,
  assistBusy: false,
  assistError: null,
};

export const useWorkbench = create<WorkbenchState>((set) => ({
  ...initial,

  loadDocument: (doc) =>
    set({
      documentId: doc.document_id,
      lang: doc.lang,
      paragraphs: doc.paragraphs,
      suggestedTerms: doc.suggested_terms,
      summary: doc.summary,
      activeParagraphId: doc.paragraphs[0]?.paragraph_id ?? null,
      deconstruct: null,
      restructure: null,
      assistError: null,
    }),

  setActiveParagraph: (id) =>
    // 切段时清掉上一段的协同结果：留着会让用户以为槽位属于当前段
    set({ activeParagraphId: id, deconstruct: null, restructure: null, assistError: null }),

  replaceParagraph: (view) =>
    set((state) => ({
      paragraphs: state.paragraphs.map((p) =>
        p.paragraph_id === view.paragraph_id ? view : p,
      ),
    })),

  markRecomputing: (id, busy) =>
    set((state) => {
      const next = new Set(state.recomputingIds);
      if (busy) next.add(id);
      else next.delete(id);
      return { recomputingIds: next };
    }),

  toggleTerm: (term) =>
    set((state) => ({
      lockedTerms: state.lockedTerms.includes(term)
        ? state.lockedTerms.filter((t) => t !== term)
        : [...state.lockedTerms, term],
    })),

  setLockedTerms: (terms) => set({ lockedTerms: terms }),

  setDeconstruct: (result) => set({ deconstruct: result, restructure: null }),

  updateSlotValue: (slotId, value) =>
    set((state) => {
      if (!state.deconstruct) return {};
      const slots: SlotCard[] = state.deconstruct.slots.map((s) =>
        s.slot_id === slotId ? { ...s, value } : s,
      );
      return { deconstruct: { ...state.deconstruct, slots } };
    }),

  setRestructure: (result) => set({ restructure: result }),
  setAssistBusy: (busy) => set({ assistBusy: busy }),
  setAssistError: (message) => set({ assistError: message }),
  reset: () => set({ ...initial, recomputingIds: new Set<string>() }),
}));
