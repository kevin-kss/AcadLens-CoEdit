/**
 * 后端 API 客户端。
 *
 * 所有请求都走这里，不在组件里直接 fetch —— 错误处理与 base URL 只该有一处。
 *
 * 503 单独处理：模型不可用时后端返回 503 并带原因，前端要据此提示「模型未配置」，
 * 而不是当成普通网络错误。两者对用户的含义完全不同。
 */

import type {
  DeconstructResult,
  DocumentResponse,
  DraftStrategy,
  Lang,
  ParagraphView,
  RestructureResult,
  SlotCard,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** 模型服务不可用。与普通网络错误区分开，提示文案不同。 */
  get isModelUnavailable(): boolean {
    return this.status === 503;
  }
}

function baseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl()}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch {
    // fetch 只在网络层失败时抛错（后端没启动、跨域被拦）
    throw new ApiError("无法连接后端服务，请确认它已启动。", 0);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // 响应体不是 JSON，保留 statusText
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export function parseDocument(text: string, lang: Lang): Promise<DocumentResponse> {
  return request<DocumentResponse>("/api/v1/documents/parse", {
    method: "POST",
    body: JSON.stringify({ text, lang }),
  });
}

export function getDocument(documentId: string): Promise<DocumentResponse> {
  return request<DocumentResponse>(`/api/v1/documents/${documentId}`);
}

export function updateParagraph(
  documentId: string,
  paragraphId: string,
  maskedText: string,
): Promise<ParagraphView> {
  return request<ParagraphView>(
    `/api/v1/documents/${documentId}/paragraphs/${paragraphId}`,
    { method: "PUT", body: JSON.stringify({ masked_text: maskedText }) },
  );
}

export function setLockedTerms(
  documentId: string,
  terms: string[],
): Promise<DocumentResponse> {
  return request<DocumentResponse>(`/api/v1/documents/${documentId}/terms`, {
    method: "PUT",
    body: JSON.stringify({ terms }),
  });
}

export function deconstruct(
  documentId: string,
  paragraphId: string,
): Promise<DeconstructResult> {
  return request<DeconstructResult>(
    `/api/v1/assist/deconstruct?document_id=${documentId}`,
    { method: "POST", body: JSON.stringify({ paragraph_id: paragraphId }) },
  );
}

export function restructure(
  documentId: string,
  paragraphId: string,
  slots: SlotCard[],
  strategies?: DraftStrategy[],
): Promise<RestructureResult> {
  return request<RestructureResult>(
    `/api/v1/assist/restructure?document_id=${documentId}`,
    {
      method: "POST",
      body: JSON.stringify({
        paragraph_id: paragraphId,
        slots,
        strategies: strategies ?? null,
      }),
    },
  );
}
