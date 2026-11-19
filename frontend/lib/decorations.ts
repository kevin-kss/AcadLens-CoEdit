/**
 * 把后端的字符偏移映射成 ProseMirror 的装饰层。
 *
 * 坐标系差异是这里唯一的难点：
 * - 后端给的是**单段纯文本内**的 Unicode 码点偏移，从 0 开始；
 * - ProseMirror 的位置是**整篇文档**的扁平坐标，从 0 开始，但每个节点的开始
 *   与结束各占 1 个位置。段落内第一个字符的位置是 paragraphStart + 1。
 *
 * 因此 offset -> PM 位置 = 段落节点起始位置 + 1 + offset。段落里若有 inline
 * 标记（加粗等）不影响这个换算，因为文本偏移与 PM 的文本位置在段落内是线性的。
 *
 * 注意：JavaScript 字符串用 UTF-16 码元索引，Python 用码点索引。对 BMP 内的
 * 中英文两者一致；但 emoji 等代理对会让二者错位。学术文本里几乎不会出现，
 * 这里用 Array.from 的码点长度做一次校验，不一致时跳过该段的装饰而非画错。
 */

import { Decoration, DecorationSet } from "@tiptap/pm/view";
import type { Node as PMNode } from "@tiptap/pm/model";
import type { IssueType, ParagraphView } from "./types";

/** 每类问题对应的 CSS 类，样式在 globals.css 里定义。 */
const ISSUE_CLASS: Record<IssueType, string> = {
  ai_connector: "issue-connector",
  hat_boot_cliche: "issue-hatboot",
  low_info_density: "issue-lowinfo",
  uniform_length: "issue-uniform",
  syntactic_parallel: "issue-parallel",
  high_curvature: "issue-curvature",
};

/** 段落级问题：整句/整段标记，用更淡的背景而非下划线，避免与词级下划线混淆。 */
const PARAGRAPH_LEVEL: ReadonlySet<IssueType> = new Set<IssueType>([
  "low_info_density",
  "uniform_length",
  "syntactic_parallel",
]);

export interface ParagraphPosition {
  paragraphId: string;
  /** 段落节点在文档中的起始位置（节点自身的位置，非内容起点）。 */
  pos: number;
  node: PMNode;
}

/**
 * 按文档顺序取出所有顶层段落节点的位置。
 *
 * 依赖「编辑器里的段落顺序与后端 paragraphs 数组顺序一致」这一前提：
 * 工作台是整篇导入后渲染的，用户不能增删段落，只能改段内文本。
 */
export function collectParagraphPositions(
  doc: PMNode,
  paragraphIds: string[],
): ParagraphPosition[] {
  const out: ParagraphPosition[] = [];
  let index = 0;
  doc.forEach((node, offset) => {
    if (node.type.name !== "paragraph") return;
    const paragraphId = paragraphIds[index];
    index += 1;
    if (paragraphId === undefined) return;
    out.push({ paragraphId, pos: offset, node });
  });
  return out;
}

/** 码元长度与码点长度不一致说明段内有代理对，偏移换算不可靠。 */
function hasAstralChars(text: string): boolean {
  return text.length !== Array.from(text).length;
}

export function buildDecorations(
  doc: PMNode,
  paragraphs: ParagraphView[],
): DecorationSet {
  const ids = paragraphs.map((p) => p.paragraph_id);
  const positions = collectParagraphPositions(doc, ids);
  const byId = new Map(paragraphs.map((p) => [p.paragraph_id, p]));
  const decorations: Decoration[] = [];

  for (const { paragraphId, pos, node } of positions) {
    const view = byId.get(paragraphId);
    if (!view) continue;

    const nodeText = node.textContent;
    // 编辑器内容与后端诊断所依据的文本不一致时（用户刚改完、重算还没回来），
    // 旧偏移会把下划线画到错位置。此时直接跳过该段，宁可暂时没下划线。
    if (nodeText !== view.masked_text) continue;
    if (hasAstralChars(nodeText)) continue;

    const contentStart = pos + 1;
    for (const issue of view.diagnosis.span_issues) {
      const from = contentStart + issue.start_char;
      const to = contentStart + issue.end_char;
      if (from >= to) continue;
      if (to > contentStart + nodeText.length) continue;

      const className = ISSUE_CLASS[issue.issue_type] ?? "issue-connector";
      const level = PARAGRAPH_LEVEL.has(issue.issue_type) ? "issue-block" : "issue-inline";
      decorations.push(
        Decoration.inline(from, to, {
          class: `issue ${level} ${className}`,
          // title 让鼠标悬停即可看到病因，不必先点开右栏
          title: issue.message,
          "data-issue-type": issue.issue_type,
        }),
      );
    }
  }

  return DecorationSet.create(doc, decorations);
}
