import { upload } from "@vercel/blob/client";

const BASE = process.env.NEXT_PUBLIC_API_URL
  ?? (process.env.NODE_ENV === "production" ? "/_/backend" : "http://localhost:8000");

export async function uploadToBlob(file: File): Promise<string> {
  const dot = file.name.lastIndexOf(".");
  const base = dot > 0 ? file.name.slice(0, dot) : file.name;
  const ext = dot > 0 ? file.name.slice(dot) : "";
  const uniqueName = `${base}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}${ext}`;

  const blob = await upload(uniqueName, file, {
    access: "public",
    handleUploadUrl: "/api/upload",
  });
  return blob.url;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }));
    const traceId = res.headers.get("x-trace-id");
    const message = err.message ?? res.statusText;
    throw new Error(traceId ? `${message} (trace: ${traceId})` : message);
  }
  return res.json();
}

// ── Session ──────────────────────────────────────────────────────────────────

export async function createSession(): Promise<{ session_id: string }> {
  return req("/session", { method: "POST", body: "{}" });
}

export async function getThreads(sessionId: string): Promise<{ threads: Thread[] }> {
  return req(`/session/${sessionId}/threads`);
}

export async function createThread(sessionId: string, title?: string): Promise<Thread> {
  return req(`/session/${sessionId}/threads`, {
    method: "POST",
    body: JSON.stringify({ title: title ?? "" }),
  });
}

export async function getHistory(sessionId: string): Promise<{ history: ResolvedIssue[] }> {
  return req(`/session/${sessionId}/history`);
}

// ── Bikes ─────────────────────────────────────────────────────────────────────

export async function getLibrary(): Promise<{ bikes: BikeEntry[] }> {
  return req("/bikes/library");
}

export async function getStarters(documentId: string): Promise<{ starters: string[] }> {
  return req(`/bikes/${documentId}/starters`);
}

// ── Ingestion ─────────────────────────────────────────────────────────────────

export async function extractMeta(blobUrl: string): Promise<MetaExtractResult> {
  const form = new FormData();
  form.append("blob_url", blobUrl);
  const res = await fetch(`${BASE}/ingest/extract-meta`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    let message = res.statusText || `HTTP ${res.status}`;
    try { message = JSON.parse(body).message ?? message; } catch { if (body) message = `${res.status}: ${body.slice(0, 120)}`; }
    const traceId = res.headers.get("x-trace-id");
    if (traceId) message = `${message} (trace: ${traceId})`;
    throw new Error(message);
  }
  return res.json();
}

export async function ingestPdf(
  blobUrl: string,
  meta: { bike_brand: string; bike_model: string; bike_year: string; manual_type: string; save_to_library?: boolean }
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("blob_url", blobUrl);
  form.append("brand", meta.bike_brand);
  form.append("model", meta.bike_model);
  form.append("year", meta.bike_year);
  form.append("manual_type", meta.manual_type);
  form.append("save_to_library", meta.save_to_library ? "true" : "false");
  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    let message = res.statusText || `HTTP ${res.status}`;
    try { message = JSON.parse(body).message ?? message; } catch { if (body) message = `${res.status}: ${body.slice(0, 120)}`; }
    const traceId = res.headers.get("x-trace-id");
    if (traceId) message = `${message} (trace: ${traceId})`;
    throw new Error(message);
  }
  return res.json();
}

export async function getIngestStatus(jobId: string): Promise<IngestStatus> {
  return req(`/ingest/status/${jobId}`);
}

// ── Query ─────────────────────────────────────────────────────────────────────

export async function query(params: QueryParams): Promise<QueryResponse> {
  return req("/query", { method: "POST", body: JSON.stringify(params) });
}

// ── Voice ──────────────────────────────────────────────────────────────────────

export async function transcribeVoice(
  audio: Blob | File,
  languageHint: string,
  sessionId?: string
): Promise<STTResponse> {
  const form = new FormData();
  const filename = audio instanceof File ? audio.name : "recording.webm";
  form.append("audio", audio, filename);
  if (languageHint) form.append("language_hint", languageHint);
  if (sessionId) form.append("session_id", sessionId);
  const res = await fetch(`${BASE}/input/voice`, { method: "POST", body: form });
  if (!res.ok) {
    const msg = (await res.json()).message ?? "STT failed";
    const traceId = res.headers.get("x-trace-id");
    throw new Error(traceId ? `${msg} (trace: ${traceId})` : msg);
  }
  return res.json();
}

// ── Image ─────────────────────────────────────────────────────────────────────

export async function describeImage(file: File): Promise<ImageResponse> {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${BASE}/input/image`, { method: "POST", body: form });
  if (!res.ok) {
    const msg = (await res.json()).message ?? "Image failed";
    const traceId = res.headers.get("x-trace-id");
    throw new Error(traceId ? `${msg} (trace: ${traceId})` : msg);
  }
  return res.json();
}

// ── TTS ───────────────────────────────────────────────────────────────────────

export async function synthesizeSpeech(text: string, language: string): Promise<TTSResult> {
  return req("/output/tts", { method: "POST", body: JSON.stringify({ text, language }) });
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Thread {
  thread_id: string;
  title: string;
  created_at: string;
}

export interface ResolvedIssue {
  thread_id: string;
  title: string;
  created_at: string;
}

export interface BikeEntry {
  document_id: string;
  bike_brand: string;
  bike_model: string;
  bike_year: string;
  manual_type: string;
  total_chunks: number;
  ingestion_timestamp: string;
}

export interface IngestStatus {
  job_id: string;
  status: "pending" | "processing" | "complete" | "error";
  progress_pct: number;
  message: string;
  document_id?: string;
}

export interface Citation {
  section_number: string;
  section_title: string;
  page_number: number;
}

export interface QueryParams {
  text: string;
  session_id: string;
  document_id: string;
  thread_id: string;
  transcript?: string;
  image_description?: string;
  voice_initiated?: boolean;
}

export interface QueryResponse {
  answer_text: string;
  spoken_summary: string;
  citations: Citation[];
  severity_label: string;
  confidence: string;
  suggested_followups: string[];
  intent: string;
  language: string;
  context_confidence: string;
  session_id: string;
  thread_id: string;
  document_id: string;
  tts?: { mocked: boolean; engine: string; text: string } | null;
}

export interface STTResponse {
  transcript: string;
  language: string;
  confidence: number;
  engine: string;
  needs_retry: boolean;
}

export interface ImageResponse {
  description: string;
  technical_terms: string[];
}

export interface TTSResult {
  mocked: boolean;
  text: string;
  engine: string;
}

export interface MetaExtractResult {
  bike_brand: string;
  bike_model: string;
  bike_year: string;
  manual_type: string;
  confidence: string;
}
