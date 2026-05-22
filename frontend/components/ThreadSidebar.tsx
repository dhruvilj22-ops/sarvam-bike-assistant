"use client";
import { useState } from "react";
import { MessageSquare, Plus, History, Home, ChevronRight, Clock, CheckCircle2 } from "lucide-react";
import type { Thread, ResolvedIssue } from "@/lib/api";

interface ThreadSidebarProps {
  threads: Thread[];
  activeThreadId: string;
  onSelectThread: (id: string) => void;
  onNewThread: () => void;
  history: ResolvedIssue[];
  bikeName: string;
  onHome: () => void;
}

export default function ThreadSidebar({
  threads,
  activeThreadId,
  onSelectThread,
  onNewThread,
  history,
  bikeName,
  onHome,
}: ThreadSidebarProps) {
  const [showHistory, setShowHistory] = useState(false);

  return (
    <aside className="w-64 flex flex-col border-r border-slate-200 bg-white h-full">
      {/* Header */}
      <div className="p-4 border-b border-slate-100">
        <button
          onClick={onHome}
          className="flex items-center gap-2 text-xs text-slate-500 hover:text-blue-600 mb-3 transition-colors"
        >
          <Home size={12} /> Change bike
        </button>
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg sarvam-gradient flex items-center justify-center shrink-0">
            <span className="text-white text-xs font-bold">{bikeName[0]}</span>
          </div>
          <div className="min-w-0">
            <p className="text-xs font-semibold text-slate-900 truncate">{bikeName}</p>
            <p className="text-[10px] text-slate-400">Service Manual</p>
          </div>
        </div>
      </div>

      {/* Active threads */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        <div className="flex items-center justify-between px-2 mb-2">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
            Issues
          </span>
          <button
            onClick={onNewThread}
            className="p-1 rounded-lg hover:bg-blue-50 text-slate-400 hover:text-blue-600 transition-colors"
            title="New issue thread"
          >
            <Plus size={13} />
          </button>
        </div>

        {threads.length === 0 ? (
          <p className="text-xs text-slate-400 text-center py-4">No active issues</p>
        ) : (
          threads.map((t) => (
            <ThreadItem
              key={t.thread_id}
              thread={t}
              active={t.thread_id === activeThreadId}
              onClick={() => onSelectThread(t.thread_id)}
            />
          ))
        )}
      </div>

      {/* History section */}
      <div className="border-t border-slate-100 p-3">
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg
                     text-xs font-medium text-slate-500 hover:text-slate-700 hover:bg-slate-50 transition-all"
        >
          <span className="flex items-center gap-1.5"><History size={12} /> Resolved Issues</span>
          <ChevronRight
            size={12}
            className={`transition-transform ${showHistory ? "rotate-90" : ""}`}
          />
        </button>

        {showHistory && (
          <div className="mt-2 space-y-1">
            {history.length === 0 ? (
              <p className="text-[11px] text-slate-400 px-2 py-2">No resolved issues yet</p>
            ) : (
              history.map((h) => (
                <div key={h.thread_id} className="flex items-start gap-2 px-2 py-2 rounded-lg bg-slate-50">
                  <CheckCircle2 size={12} className="text-green-500 mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <p className="text-xs text-slate-700 truncate">{h.title || "Resolved issue"}</p>
                    <p className="text-[10px] text-slate-400">{formatDate(h.created_at)}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

function ThreadItem({ thread, active, onClick }: { thread: Thread; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-start gap-2.5 px-3 py-2.5 rounded-xl text-left transition-all
        ${active
          ? "bg-blue-50 border border-blue-200 text-blue-700"
          : "text-slate-600 hover:bg-slate-50 hover:text-slate-800"
        }`}
    >
      <MessageSquare size={13} className={`mt-0.5 shrink-0 ${active ? "text-blue-500" : "text-slate-400"}`} />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium truncate">{thread.title || "New issue"}</p>
        <p className="text-[10px] text-slate-400 flex items-center gap-1 mt-0.5">
          <Clock size={9} /> {formatDate(thread.created_at)}
        </p>
      </div>
    </button>
  );
}

function formatDate(iso: string) {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}
