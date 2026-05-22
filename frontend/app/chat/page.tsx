"use client";
import { useState, useEffect, useRef, useCallback, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, AlertCircle, Bike, Menu, X as XIcon } from "lucide-react";
import MessageCard, { type Message } from "@/components/MessageCard";
import InputBar from "@/components/InputBar";
import ThreadSidebar from "@/components/ThreadSidebar";
import {
  query, getThreads, createThread, getHistory, getStarters,
  type Thread, type ResolvedIssue,
} from "@/lib/api";

function ChatInner() {
  const router = useRouter();
  const params = useSearchParams();
  const sessionId = params.get("session") ?? "";
  const documentId = params.get("doc") ?? "";
  const bikeName = params.get("bike") ?? documentId;

  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThread, setActiveThread] = useState<Thread | null>(null);
  const [history, setHistory] = useState<ResolvedIssue[]>([]);
  const [messages, setMessages] = useState<Record<string, Message[]>>({});
  const [loading, setLoading] = useState(false);
  const [initError, setInitError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [language, setLanguage] = useState("en");
  const bottomRef = useRef<HTMLDivElement>(null);

  // Initialise: load threads and create first one if empty
  useEffect(() => {
    if (!sessionId || !documentId) { router.push("/"); return; }
    async function init() {
      try {
        const [t, h] = await Promise.all([
          getThreads(sessionId),
          getHistory(sessionId),
        ]);
        const existingThreads = t.threads ?? [];
        setThreads(existingThreads);
        setHistory(h.history ?? []);
        if (existingThreads.length > 0) {
          setActiveThread(existingThreads[0]);
        } else {
          const newThread = await createThread(sessionId, "New issue");
          setThreads([newThread]);
          setActiveThread(newThread);
        }
      } catch {
        setInitError("Failed to load session. Please go back and try again.");
      }
    }
    init();
  }, [sessionId, documentId, router]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeThread?.thread_id]);

  const currentMessages = (activeThread ? messages[activeThread.thread_id] : []) ?? [];

  const handleNewThread = useCallback(async () => {
    try {
      const t = await createThread(sessionId, "New issue");
      setThreads((prev) => [t, ...prev]);
      setActiveThread(t);
      setSidebarOpen(false);
    } catch {/* ignore */}
  }, [sessionId]);

  const handleSend = useCallback(async (params: {
    text: string;
    transcript?: string;
    imageDescription?: string;
    imagePreview?: string;
    voiceInitiated?: boolean;
  }) => {
    if (!activeThread || loading) return;
    const tid = activeThread.thread_id;

    // Compose display text for user bubble
    const displayText = params.transcript
      ? params.transcript + (params.text ? ` — ${params.text}` : "")
      : params.text;

    const userMsg: Message = {
      role: "user",
      text: displayText,
      imagePreview: params.imagePreview,
      timestamp: new Date(),
    };

    setMessages((prev) => ({
      ...prev,
      [tid]: [...(prev[tid] ?? []), userMsg],
    }));

    // Update thread title from first message
    if (!(messages[tid]?.length) && displayText) {
      const title = displayText.slice(0, 50);
      setThreads((prev) =>
        prev.map((t) => (t.thread_id === tid ? { ...t, title } : t))
      );
      if (activeThread.title === "New issue") {
        setActiveThread((t) => t ? { ...t, title } : t);
      }
    }

    setLoading(true);
    try {
      const resp = await query({
        text: params.text || params.transcript || "",
        session_id: sessionId,
        document_id: documentId,
        thread_id: tid,
        transcript: params.transcript,
        image_description: params.imageDescription,
        voice_initiated: params.voiceInitiated,
      });

      // Detect language from response
      if (resp.language) setLanguage(resp.language);

      const assistantMsg: Message = {
        role: "assistant",
        text: resp.answer_text,
        response: resp,
        timestamp: new Date(),
      };

      setMessages((prev) => ({
        ...prev,
        [tid]: [...(prev[tid] ?? []), assistantMsg],
      }));
    } catch (err: unknown) {
      const errMsg: Message = {
        role: "assistant",
        text: `Sorry, something went wrong: ${err instanceof Error ? err.message : "Unknown error"}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages((prev) => ({ ...prev, [tid]: [...(prev[tid] ?? []), errMsg] }));
    } finally {
      setLoading(false);
    }
  }, [activeThread, loading, messages, sessionId, documentId]);

  const handleFollowup = useCallback((q: string) => {
    handleSend({ text: q });
  }, [handleSend]);

  if (initError) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-slate-500">
        <AlertCircle size={24} className="text-red-400" />
        <p className="text-sm">{initError}</p>
        <button onClick={() => router.push("/")} className="text-sm text-blue-600 hover:underline">
          Go back
        </button>
      </div>
    );
  }

  if (!activeThread) {
    return (
      <div className="flex items-center justify-center h-full gap-2 text-slate-400">
        <Loader2 size={18} className="animate-spin" />
        <span className="text-sm">Loading session...</span>
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Mobile sidebar overlay ───────────────────────── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-30 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ──────────────────────────────────────── */}
      <div
        className={`fixed md:relative inset-y-0 left-0 z-40 md:z-auto
          transform transition-transform duration-200
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
          flex flex-col h-full`}
      >
        <ThreadSidebar
          threads={threads}
          activeThreadId={activeThread.thread_id}
          onSelectThread={(id) => {
            const t = threads.find((th) => th.thread_id === id);
            if (t) { setActiveThread(t); setSidebarOpen(false); }
          }}
          onNewThread={handleNewThread}
          history={history}
          bikeName={bikeName}
          onHome={() => router.push("/")}
        />
      </div>

      {/* ── Main chat ────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center gap-3 px-4 h-14 border-b border-slate-200 bg-white/90 backdrop-blur-sm shrink-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="md:hidden p-2 rounded-lg hover:bg-slate-100 text-slate-500"
          >
            {sidebarOpen ? <XIcon size={18} /> : <Menu size={18} />}
          </button>
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-7 h-7 rounded-lg sarvam-gradient flex items-center justify-center shrink-0">
              <Bike size={13} className="text-white" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold text-slate-900 truncate">{activeThread.title || "New issue"}</p>
              <p className="text-xs text-slate-400 truncate">{bikeName}</p>
            </div>
          </div>
        </header>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {currentMessages.length === 0 ? (
            <WelcomePrompt bikeName={bikeName} documentId={documentId} onPrompt={(q) => handleSend({ text: q })} />
          ) : (
            currentMessages.map((msg, i) => (
              <MessageCard key={i} message={msg} onFollowup={handleFollowup} />
            ))
          )}

          {/* Loading indicator */}
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full sarvam-gradient flex items-center justify-center shrink-0">
                <Loader2 size={14} className="text-white animate-spin" />
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                <div className="flex gap-1 items-center">
                  <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div className="shrink-0 px-4 pb-4 pt-2 border-t border-slate-200 bg-white/90 backdrop-blur-sm">
          <InputBar
            onSubmit={handleSend}
            disabled={loading}
            sessionId={sessionId}
            language={language}
          />
        </div>
      </div>
    </div>
  );
}

// ── Starter prompts ────────────────────────────────────────────────────────────

const FALLBACK_STARTERS = [
  "What does white smoke from the exhaust mean?",
  "How do I check and change the engine oil?",
  "What is the valve clearance specification?",
  "My bike is making a knocking sound — what could it be?",
];

function WelcomePrompt({
  bikeName,
  documentId,
  onPrompt,
}: {
  bikeName: string;
  documentId: string;
  onPrompt: (q: string) => void;
}) {
  const [starters, setStarters] = useState<string[]>(FALLBACK_STARTERS);

  useEffect(() => {
    if (!documentId) return;
    getStarters(documentId)
      .then((r) => { if (r.starters?.length) setStarters(r.starters); })
      .catch(() => {/* keep fallbacks */});
  }, [documentId]);

  return (
    <div className="flex flex-col items-center text-center py-8 max-w-md mx-auto">
      <div className="w-16 h-16 rounded-2xl sarvam-gradient flex items-center justify-center mb-5 shadow-lg">
        <Bike size={28} className="text-white" />
      </div>
      <h2 className="text-xl font-bold text-slate-900 mb-2">Ask about your bike</h2>
      <p className="text-sm text-slate-500 mb-6">
        Every answer comes directly from your <span className="font-medium text-slate-700">{bikeName}</span> manual.
        You can type, speak, or attach a photo.
      </p>
      <div className="w-full space-y-2">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-3">Suggested questions</p>
        {starters.map((q) => (
          <button
            key={q}
            onClick={() => onPrompt(q)}
            className="w-full text-left px-4 py-3 rounded-xl border border-slate-200 bg-white
                       text-sm text-slate-700 hover:border-blue-400 hover:bg-blue-50 hover:text-blue-700
                       transition-all shadow-sm flex items-center justify-between group"
          >
            {q}
            <span className="text-slate-300 group-hover:text-blue-400 text-xs ml-2 shrink-0">→</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-full gap-2 text-slate-400">
        <Loader2 size={18} className="animate-spin" />
      </div>
    }>
      <ChatInner />
    </Suspense>
  );
}
