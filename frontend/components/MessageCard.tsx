"use client";
import { useState } from "react";
import { Play, Pause, Volume2, AlertTriangle, Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import SeverityBadge from "./SeverityBadge";
import CitationBlock from "./CitationBlock";
import FollowupChips from "./FollowupChips";
import type { QueryResponse } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  text: string;
  response?: QueryResponse;
  imagePreview?: string;
  timestamp: Date;
}

export default function MessageCard({
  message,
  onFollowup,
}: {
  message: Message;
  onFollowup: (q: string) => void;
}) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5
          ${isUser ? "bg-orange-100 text-orange-600" : "sarvam-gradient text-white"}`}
      >
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>

      <div className={`max-w-[80%] space-y-2 ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        {/* Image preview if attached */}
        {message.imagePreview && (
          <div className="rounded-xl overflow-hidden border border-slate-200">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={message.imagePreview} alt="Attached" className="max-h-40 object-cover" />
          </div>
        )}

        {/* Bubble — hidden for assistant messages with a structured response (AssistantExtras renders the answer) */}
        {(isUser || !message.response) && (
          <div
            className={`px-4 py-3 rounded-2xl text-sm leading-relaxed
              ${isUser
                ? "bg-blue-600 text-white rounded-tr-sm"
                : "bg-white border border-slate-200 text-slate-800 rounded-tl-sm shadow-sm"
              }`}
          >
            {message.text}
          </div>
        )}

        {/* Assistant response extras */}
        {!isUser && message.response && (
          <AssistantExtras response={message.response} onFollowup={onFollowup} />
        )}

        {/* Timestamp */}
        <p className={`text-[10px] text-slate-400 px-1 ${isUser ? "text-right" : ""}`}>
          {message.timestamp.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
        </p>
      </div>
    </div>
  );
}

function AssistantExtras({
  response,
  onFollowup,
}: {
  response: QueryResponse;
  onFollowup: (q: string) => void;
}) {
  return (
    <div className="w-full bg-white border border-slate-200 rounded-2xl rounded-tl-sm shadow-sm overflow-hidden">
      {/* Severity + confidence header */}
      {(response.severity_label || response.context_confidence === "low") && (
        <div className="flex items-center gap-2 px-4 pt-3 flex-wrap">
          {response.severity_label && <SeverityBadge label={response.severity_label} />}
          {response.context_confidence === "low" && (
            <span className="inline-flex items-center gap-1 text-xs text-amber-600 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
              <AlertTriangle size={10} /> Low confidence — verify with mechanic
            </span>
          )}
        </div>
      )}

      {/* Answer */}
      <div className="px-4 py-3 text-sm text-slate-800 leading-relaxed markdown-answer">
        <ReactMarkdown
          components={{
            p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
            ol: ({ children }) => <ol className="list-decimal list-outside pl-5 space-y-1 mb-2">{children}</ol>,
            ul: ({ children }) => <ul className="list-disc list-outside pl-5 space-y-1 mb-2">{children}</ul>,
            li: ({ children }) => <li className="leading-relaxed">{children}</li>,
            strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
            em: ({ children }) => <em className="italic text-slate-600">{children}</em>,
          }}
        >
          {response.answer_text}
        </ReactMarkdown>
      </div>

      {/* TTS player */}
      {response.tts && <TTSPlayer text={response.tts.text} />}

      {/* Citations */}
      {response.citations?.length > 0 && (
        <div className="px-4 pb-3">
          <CitationBlock citations={response.citations} />
        </div>
      )}

      {/* Followups */}
      {response.suggested_followups?.length > 0 && (
        <div className="px-4 pb-4">
          <FollowupChips followups={response.suggested_followups} onSelect={onFollowup} />
        </div>
      )}
    </div>
  );
}

function TTSPlayer({ text }: { text: string }) {
  const [playing, setPlaying] = useState(false);

  function speak() {
    if (!("speechSynthesis" in window)) return;
    if (playing) {
      speechSynthesis.cancel();
      setPlaying(false);
      return;
    }
    const utter = new SpeechSynthesisUtterance(text);
    utter.onend = () => setPlaying(false);
    utter.onerror = () => setPlaying(false);
    setPlaying(true);
    speechSynthesis.speak(utter);
  }

  return (
    <div className="mx-4 mb-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-50 border border-blue-100">
      <Volume2 size={13} className="text-blue-500 shrink-0" />
      <p className="text-xs text-blue-600 flex-1 line-clamp-1 italic">{text}</p>
      <button
        onClick={speak}
        className={`w-7 h-7 rounded-full flex items-center justify-center transition-all
          ${playing
            ? "bg-orange-500 text-white hover:bg-orange-600"
            : "bg-blue-600 text-white hover:bg-blue-700"
          }`}
      >
        {playing ? <Pause size={11} /> : <Play size={11} />}
      </button>
    </div>
  );
}

export type { Message };
