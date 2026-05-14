import type { Message, PdfImage, UsageMode } from "./types";

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) return res.json() as Promise<T>;
  let detail: string | undefined;
  try {
    const body = await res.json();
    detail = body?.detail;
  } catch {
    // ignore parse errors
  }
  throw new Error(detail ?? `HTTP ${res.status} ${res.statusText}`);
}

export async function ingestPdf(
  file: File
): Promise<{ ok: boolean; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/ingest", { method: "POST", body: form });
  return handleResponse(res);
}

export async function queryRag(args: {
  question: string;
  conversationHistory: Message[];
  provider: string;
  modelId: string;
  apiKey: string;
  usageMode: UsageMode;
}): Promise<{ answer: string; conversationHistory: Message[] }> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: args.question,
      conversation_history: args.conversationHistory,
      provider: args.provider,
      model_id: args.modelId,
      api_key: args.apiKey,
      usage_mode: args.usageMode,
    }),
  });
  const data = await handleResponse<{
    answer: string;
    conversation_history: Message[];
  }>(res);
  return {
    answer: data.answer,
    conversationHistory: data.conversation_history,
  };
}

export async function summarize(args: {
  provider: string;
  modelId: string;
  apiKey: string;
  usageMode: UsageMode;
}): Promise<{ summary: string }> {
  const res = await fetch("/api/summarize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: args.provider,
      model_id: args.modelId,
      api_key: args.apiKey,
      usage_mode: args.usageMode,
    }),
  });
  return handleResponse(res);
}

export async function getEnvConfig(): Promise<{
  provider: string;
  model_id: string;
  has_key: boolean;
}> {
  const res = await fetch("/api/env-config");
  return handleResponse(res);
}

export async function ingestDemo(): Promise<{ ok: boolean; filename: string }> {
  const res = await fetch("/api/ingest-demo", { method: "POST" });
  return handleResponse(res);
}

export async function extractImages(
  file: File
): Promise<{ images: PdfImage[] }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/images", { method: "POST", body: form });
  return handleResponse(res);
}
