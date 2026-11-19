/**
 * 后端契约的 TypeScript 镜像。
 *
 * 字段名与 backend/app/schemas/ 下的 Pydantic 模型逐一对应 —— 诊断结果直接
 * 驱动编辑器的装饰层坐标，字段错位会让下划线画到错误的位置。改后端 schema
 * 时这里必须同步改。
 */

export type Lang = "zh" | "en";

export type RiskLevel = "low" | "medium" | "high";

export type IssueType =
  | "ai_connector"
  | "hat_boot_cliche"
  | "low_info_density"
  | "uniform_length"
  | "syntactic_parallel"
  | "high_curvature";

/**
 * 一处可定位的问题。
 *
 * start_char / end_char 是相对于**带掩码段落文本**的 Unicode 码点偏移。
 * 映射到 ProseMirror 位置时必须考虑：PM 的位置从 1 开始，且节点边界占位。
 */
export interface TextSpanIssue {
  issue_type: IssueType;
  start_char: number;
  end_char: number;
  matched_text: string;
  message: string;
}

export interface SentenceMetric {
  sent_index: number;
  text: string;
  start_char: number;
  clean_char_len: number;
  ppl: number | null;
  curvature: number | null;
  dep_skeleton: string;
  /** dep（真实依存）| pos（无模型降级）。决定分数的可信度，须对用户可见。 */
  skeleton_source: string;
  fluff_ratio: number;
  concrete_anchors: number;
  sentence_risk: number;
}

export interface TrackScores {
  syntactic: number;
  lexicon: number;
  curvature: number | null;
}

export interface ParagraphDiagnosis {
  paragraph_id: string;
  lang: Lang;
  mode: "dual_track" | "tri_track";
  burstiness_cv: number;
  track_scores: TrackScores;
  overall_risk: number;
  risk_level: RiskLevel;
  sentences: SentenceMetric[];
  span_issues: TextSpanIssue[];
}

export interface ParagraphView {
  paragraph_id: string;
  order_index: number;
  section_type: string;
  masked_text: string;
  plain_text: string;
  diagnosis: ParagraphDiagnosis;
}

export interface TermCandidate {
  term: string;
  score: number;
  frequency: number;
  word_count: number;
  nested_in: number;
  source: "auto_extracted" | "user_added";
}

export interface DocumentSummary {
  paragraph_count: number;
  mean_risk: number;
  risk_level: RiskLevel;
  high_risk_paragraph_ids: string[];
}

export interface DocumentResponse {
  document_id: string;
  lang: Lang;
  paragraphs: ParagraphView[];
  suggested_terms: TermCandidate[];
  summary: DocumentSummary;
}

// ---------------------------------------------------------------------------
// 协同重构
// ---------------------------------------------------------------------------

export type SlotKind =
  | "numeric"
  | "dataset"
  | "metric"
  | "citation"
  | "case"
  | "condition"
  | "other";

export interface FactTriple {
  subject: string;
  predicate: string;
  obj: string;
  source_quote: string;
}

export interface SlotCard {
  slot_id: string;
  kind: SlotKind;
  prompt: string;
  rationale: string;
  /** null 表示作者跳过。跳过与填空串是不同语义，不要合并。 */
  value: string | null;
}

export interface DeconstructResult {
  paragraph_id: string;
  lang: Lang;
  facts: FactTriple[];
  slots: SlotCard[];
  locked_terms: string[];
  outline: string;
}

/**
 * Stage 2 的生成策略。只保留一种：用户要的是能直接复制使用的成稿，
 * 并列多版等于把系统该做的判断推给用户。保留联合类型形态以便将来扩展。
 */
export type DraftStrategy = "plain";

export type GuardViolationType =
  | "mask_token_missing"
  | "mask_token_altered"
  | "locked_term_missing";

export interface TermGuardViolation {
  violation_type: GuardViolationType;
  subject: string;
  expected_count: number;
  actual_count: number;
  message: string;
}

export interface TermGuardResult {
  passed: boolean;
  violations: TermGuardViolation[];
}

export interface DraftCandidate {
  strategy: DraftStrategy;
  masked_text: string;
  guard_result: TermGuardResult;
  risk_before: number;
  risk_after: number;
  retry_count: number;
  /** 非空表示被门禁拒绝，前端置灰展示而非隐藏。 */
  reject_reason: string;
}

export interface RestructureResult {
  paragraph_id: string;
  lang: Lang;
  risk_before: number;
  candidates: DraftCandidate[];
}
