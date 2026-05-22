"use client";
import { useState, useRef, useCallback } from "react";
import { Mic, MicOff, ImagePlus, Send, X, Loader2, AlertCircle } from "lucide-react";
import { transcribeVoice, describeImage } from "@/lib/api";

interface InputBarProps {
  onSubmit: (params: {
    text: string;
    transcript?: string;
    imageDescription?: string;
    imagePreview?: string;
    voiceInitiated?: boolean;
  }) => void;
  disabled?: boolean;
  sessionId?: string;
  language?: string;
}

export default function InputBar({ onSubmit, disabled, sessionId, language = "en" }: InputBarProps) {
  const [text, setText] = useState("");
  const [recording, setRecording] = useState(false);
  const [processingVoice, setProcessingVoice] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState("");
  const [imageDescription, setImageDescription] = useState("");
  const [processingImage, setProcessingImage] = useState(false);
  const [error, setError] = useState("");
  const [voiceInitiated, setVoiceInitiated] = useState(false);

  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const toggleRecording = useCallback(async () => {
    if (recording) {
      mediaRef.current?.stop();
      setRecording(false);
      return;
    }
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const PREFERRED = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
      const mimeType = PREFERRED.find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const actualType = mr.mimeType || mimeType || "audio/webm";
        const ext = actualType.includes("ogg") ? "ogg" : actualType.includes("mp4") ? "mp4" : "webm";
        const file = new File(chunksRef.current, `recording.${ext}`, { type: actualType });
        if (file.size < 500) {
          setError("Recording too short — speak for at least 1 second.");
          return;
        }
        setProcessingVoice(true);
        try {
          const stt = await transcribeVoice(file, language, sessionId);
          if (stt.needs_retry) {
            setError("Didn't catch that clearly — please try again or type your question.");
          } else {
            setTranscript(stt.transcript);
            setVoiceInitiated(true);
          }
        } catch {
          setError("Voice input failed — please type your question.");
        } finally {
          setProcessingVoice(false);
        }
      };
      mediaRef.current = mr;
      mr.start(100);
      setRecording(true);
    } catch {
      setError("Microphone access denied.");
    }
  }, [recording, language, sessionId]);

  async function handleImage(file: File) {
    setError("");
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImagePreview(url);
    setProcessingImage(true);
    try {
      const result = await describeImage(file);
      setImageDescription(result.description);
    } catch {
      setError("Image processing failed.");
      setImageFile(null);
      setImagePreview("");
    } finally {
      setProcessingImage(false);
    }
  }

  function clearImage() {
    setImageFile(null);
    setImagePreview("");
    setImageDescription("");
    if (imageInputRef.current) imageInputRef.current.value = "";
  }

  function clearTranscript() {
    setTranscript("");
    setVoiceInitiated(false);
  }

  function handleSubmit() {
    const hasContent = text.trim() || transcript || imageDescription;
    if (!hasContent || disabled) return;
    onSubmit({
      text: text.trim(),
      transcript: transcript || undefined,
      imageDescription: imageDescription || undefined,
      imagePreview: imagePreview || undefined,
      voiceInitiated,
    });
    setText("");
    setTranscript("");
    setVoiceInitiated(false);
    setImageFile(null);
    setImagePreview("");
    setImageDescription("");
    setError("");
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  }

  const hasContent = text.trim() || transcript || imageDescription;
  const busy = processingVoice || processingImage;

  return (
    <div className="space-y-2">
      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-xs text-red-600">
          <AlertCircle size={13} />
          {error}
          <button onClick={() => setError("")} className="ml-auto"><X size={12} /></button>
        </div>
      )}

      {/* Transcript chip */}
      {transcript && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-orange-50 border border-orange-200">
          <Mic size={13} className="text-orange-500 shrink-0" />
          <span className="text-xs text-orange-700 flex-1 italic">&ldquo;{transcript}&rdquo;</span>
          <button onClick={clearTranscript} className="text-orange-400 hover:text-orange-600">
            <X size={12} />
          </button>
        </div>
      )}

      {/* Image preview chip */}
      {imagePreview && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-50 border border-slate-200">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imagePreview} alt="" className="w-8 h-8 rounded object-cover border" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-slate-700 truncate">{imageFile?.name}</p>
            {processingImage
              ? <p className="text-xs text-slate-400">Analysing image...</p>
              : <p className="text-xs text-slate-400 truncate">{imageDescription}</p>
            }
          </div>
          {processingImage
            ? <Loader2 size={13} className="text-blue-500 animate-spin shrink-0" />
            : <button onClick={clearImage} className="text-slate-400 hover:text-slate-600 shrink-0"><X size={12} /></button>
          }
        </div>
      )}

      {/* Main input row */}
      <div className="flex items-end gap-2 p-1.5 rounded-2xl border border-slate-200 bg-white shadow-sm focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition-all">
        {/* Voice button */}
        <button
          type="button"
          onClick={toggleRecording}
          disabled={disabled || processingVoice}
          className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 transition-all
            ${recording
              ? "bg-red-500 text-white animate-pulse"
              : processingVoice
              ? "bg-slate-100 text-slate-400"
              : "bg-slate-100 text-slate-500 hover:bg-blue-100 hover:text-blue-600"
            }`}
          title={recording ? "Click to stop recording" : "Click to start recording"}
        >
          {processingVoice
            ? <Loader2 size={16} className="animate-spin" />
            : recording
            ? <MicOff size={16} />
            : <Mic size={16} />
          }
        </button>

        {/* Image button */}
        <button
          type="button"
          onClick={() => imageInputRef.current?.click()}
          disabled={disabled || !!imageFile || processingImage}
          className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0 transition-all
                     bg-slate-100 text-slate-500 hover:bg-blue-100 hover:text-blue-600
                     disabled:opacity-40 disabled:cursor-not-allowed"
          title="Attach image"
        >
          <ImagePlus size={16} />
        </button>
        <input
          ref={imageInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleImage(f); }}
        />

        {/* Text area */}
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={recording ? "Recording..." : "Ask about your bike... (Hindi or English)"}
          disabled={disabled || recording}
          rows={1}
          className="flex-1 resize-none bg-transparent text-sm text-slate-800 placeholder:text-slate-400
                     focus:outline-none py-2 px-1 max-h-32 leading-relaxed"
          style={{ overflowY: text.split("\n").length > 3 ? "auto" : "hidden" }}
        />

        {/* Send button */}
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!hasContent || disabled || busy}
          className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 transition-all
            ${hasContent && !disabled && !busy
              ? "sarvam-gradient text-white shadow-sm hover:opacity-90 active:scale-95"
              : "bg-slate-100 text-slate-300 cursor-not-allowed"
            }`}
        >
          <Send size={15} />
        </button>
      </div>

      <p className="text-center text-[10px] text-slate-400">
        {recording ? "🔴 Recording — click mic again to stop" : "Click mic to speak · Attach an image · Press Enter to send"}
      </p>
    </div>
  );
}
