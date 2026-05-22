"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  Bike, Upload, Library, ChevronRight, AlertCircle,
  CheckCircle2, Loader2, X, Wrench, Zap
} from "lucide-react";
import { getLibrary, ingestPdf, getIngestStatus, createSession, extractMeta, type BikeEntry } from "@/lib/api";

const MANUAL_TYPES = [
  { value: "service_manual", label: "Service Manual" },
  { value: "owner_manual", label: "Owner's Manual" },
  { value: "user_guide", label: "User Guide" },
];

export default function HomePage() {
  const router = useRouter();
  const [library, setLibrary] = useState<BikeEntry[]>([]);
  const [tab, setTab] = useState<"library" | "upload">("library");
  const [loading, setLoading] = useState(true);

  // upload state
  const [file, setFile] = useState<File | null>(null);
  const [brand, setBrand] = useState("");
  const [model, setModel] = useState("");
  const [year, setYear] = useState(String(new Date().getFullYear()));
  const [manualType, setManualType] = useState("service_manual");
  const [saveToLibrary, setSaveToLibrary] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [uploadDone, setUploadDone] = useState(false);
  const [uploadDocId, setUploadDocId] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getLibrary()
      .then((r) => setLibrary(r.bikes ?? []))
      .catch(() => setLibrary([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setInterval(async () => {
      try {
        const s = await getIngestStatus(jobId);
        setProgress(s.progress_pct);
        setProgressMsg(s.message);
        if (s.status === "complete") {
          clearInterval(pollRef.current!);
          setUploadDone(true);
          setUploadDocId(s.document_id ?? "");
        } else if (s.status === "error") {
          clearInterval(pollRef.current!);
          setUploadError(s.message);
          setUploading(false);
        }
      } catch {
        // keep polling
      }
    }, 2000);
    return () => clearInterval(pollRef.current!);
  }, [jobId]);

  async function startSession(documentId: string) {
    const { session_id } = await createSession();
    router.push(`/chat?session=${session_id}&doc=${documentId}`);
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !brand || !model) return;
    setUploading(true);
    setUploadError("");
    setProgress(0);
    setProgressMsg("Starting...");
    try {
      const { job_id } = await ingestPdf(file, {
        bike_brand: brand,
        bike_model: model,
        bike_year: year,
        manual_type: manualType,
        save_to_library: saveToLibrary,
      });
      setJobId(job_id);
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
    }
  }

  const uploadValid = file && brand.trim() && model.trim();

  return (
    <div className="min-h-full flex flex-col">
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg sarvam-gradient flex items-center justify-center">
              <Wrench size={15} className="text-white" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-bold text-slate-900 text-sm tracking-tight">Bike Assistant</span>
              <span className="text-[10px] font-semibold uppercase tracking-widest text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                Sarvam AI
              </span>
            </div>
          </div>
          <div className="flex items-center gap-1 text-xs text-slate-400">
            <Zap size={12} className="text-orange-400" />
            Manual-grounded answers only
          </div>
        </div>
      </header>

      {/* ── Hero ───────────────────────────────────────────── */}
      <section className="relative overflow-hidden py-16 sm:py-20">
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full opacity-[0.06] sarvam-gradient" />
          <div className="absolute -bottom-32 -left-32 w-[500px] h-[500px] rounded-full opacity-[0.04] sarvam-gradient" />
        </div>
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 text-center">
          <div className="inline-flex items-center gap-2 mb-5 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-100 text-xs font-medium text-blue-700">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
            AI-powered from your bike&apos;s manual
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-900 mb-4">
            Diagnose your bike
            <br />
            <span className="sarvam-gradient-text">from the manual</span>
          </h1>
          <p className="text-slate-500 text-lg max-w-xl mx-auto">
            Ask in Hindi or English. Get cited answers from your bike&apos;s service manual.
            No guesses — only what&apos;s in the book.
          </p>
        </div>
      </section>

      {/* ── Main card ──────────────────────────────────────── */}
      <main className="flex-1 pb-16">
        <div className="max-w-3xl mx-auto px-4 sm:px-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            {/* Tab bar */}
            <div className="flex border-b border-slate-200">
              {(["library", "upload"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={`flex-1 flex items-center justify-center gap-2 py-3.5 text-sm font-medium transition-all
                    ${tab === t
                      ? "text-blue-700 border-b-2 border-blue-600 bg-blue-50/50"
                      : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                    }`}
                >
                  {t === "library" ? <Library size={15} /> : <Upload size={15} />}
                  {t === "library" ? "Bike Library" : "Upload Manual"}
                </button>
              ))}
            </div>

            {/* ── Library tab ─────────────────────────────── */}
            {tab === "library" && (
              <div className="p-6">
                {loading ? (
                  <div className="flex items-center justify-center py-12 text-slate-400 gap-2">
                    <Loader2 size={18} className="animate-spin" />
                    <span className="text-sm">Loading library...</span>
                  </div>
                ) : library.length === 0 ? (
                  <EmptyLibrary onUpload={() => setTab("upload")} />
                ) : (
                  <div className="space-y-3">
                    <p className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-4">
                      {library.length} manual{library.length !== 1 ? "s" : ""} ready
                    </p>
                    {library.map((bike) => (
                      <BikeCard key={bike.document_id} bike={bike} onSelect={startSession} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ── Upload tab ──────────────────────────────── */}
            {tab === "upload" && (
              <div className="p-6">
                {uploadDone ? (
                  <UploadSuccess brand={brand} model={model} onStart={() => startSession(uploadDocId)} />
                ) : (
                  <form onSubmit={handleUpload} className="space-y-5">
                    {/* File drop zone */}
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className={`relative flex flex-col items-center justify-center gap-3 py-10
                        rounded-xl border-2 border-dashed cursor-pointer transition-all
                        ${file
                          ? "border-blue-400 bg-blue-50"
                          : "border-slate-200 hover:border-blue-300 hover:bg-slate-50"
                        }`}
                    >
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf"
                        className="hidden"
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (!f) return;
                          setFile(f);
                          setUploadError("");
                          setExtracting(true);
                          extractMeta(f)
                            .then((meta) => {
                              if (meta.bike_brand) setBrand(meta.bike_brand);
                              if (meta.bike_model) setModel(meta.bike_model);
                              if (meta.bike_year) setYear(meta.bike_year);
                              if (meta.manual_type) setManualType(meta.manual_type);
                            })
                            .catch(() => {/* user fills in manually */})
                            .finally(() => setExtracting(false));
                        }}
                      />
                      {file ? (
                        <>
                          <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                            <CheckCircle2 size={20} className="text-blue-600" />
                          </div>
                          <div className="text-center">
                            <p className="text-sm font-medium text-slate-900">{file.name}</p>
                            <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); setFile(null); }}
                            className="absolute top-3 right-3 p-1 rounded-full hover:bg-slate-200 text-slate-400"
                          >
                            <X size={14} />
                          </button>
                        </>
                      ) : (
                        <>
                          <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center">
                            <Upload size={18} className="text-slate-400" />
                          </div>
                          <div className="text-center">
                            <p className="text-sm font-medium text-slate-700">Drop your PDF here</p>
                            <p className="text-xs text-slate-400">Service manual, owner&apos;s manual, or user guide</p>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Bike metadata */}
                    <div className="space-y-1">
                      {extracting && (
                        <div className="flex items-center gap-1.5 text-xs text-blue-600 pb-1">
                          <Loader2 size={11} className="animate-spin" />
                          Auto-detecting from PDF...
                        </div>
                      )}
                    </div>
                    <div className={`grid grid-cols-2 gap-4 transition-opacity ${extracting ? "opacity-60 pointer-events-none" : ""}`}>
                      <LabeledInput label="Brand *" placeholder="Royal Enfield" value={brand} onChange={setBrand} />
                      <LabeledInput label="Model *" placeholder="Meteor 350" value={model} onChange={setModel} />
                      <LabeledInput label="Year" placeholder="2022" value={year} onChange={setYear} />
                      <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1.5">Manual Type</label>
                        <select
                          value={manualType}
                          onChange={(e) => setManualType(e.target.value)}
                          className="w-full px-3 py-2.5 text-sm rounded-lg border border-slate-200 bg-white
                                     focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        >
                          {MANUAL_TYPES.map((t) => (
                            <option key={t.value} value={t.value}>{t.label}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Save to library checkbox */}
                    <label className="flex items-start gap-3 p-3.5 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/40 cursor-pointer transition-all group">
                      <div className="mt-0.5 relative">
                        <input
                          type="checkbox"
                          checked={saveToLibrary}
                          onChange={(e) => setSaveToLibrary(e.target.checked)}
                          className="peer w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                        />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-slate-700 group-hover:text-blue-700 transition-colors">
                          Save to library
                        </p>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Make this manual available on the Bike Library tab for all sessions
                        </p>
                      </div>
                    </label>

                    {/* Progress bar */}
                    {uploading && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs text-slate-500">
                          <span>{progressMsg}</span>
                          <span>{progress}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className="h-full rounded-full sarvam-gradient transition-all duration-700"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {uploadError && (
                      <div className="flex items-center gap-2 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
                        <AlertCircle size={15} />
                        {uploadError}
                      </div>
                    )}

                    <button
                      type="submit"
                      disabled={!uploadValid || uploading || extracting}
                      className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2
                        transition-all ${uploadValid && !uploading && !extracting
                          ? "sarvam-gradient text-white shadow-md hover:opacity-90 active:scale-[0.99]"
                          : "bg-slate-100 text-slate-400 cursor-not-allowed"
                        }`}
                    >
                      {uploading
                        ? <><Loader2 size={15} className="animate-spin" /> Indexing manual...</>
                        : extracting
                          ? <><Loader2 size={15} className="animate-spin" /> Detecting metadata...</>
                          : <><Upload size={15} /> Index this manual</>
                      }
                    </button>

                    <p className="text-xs text-center text-slate-400">
                      Indexing runs in the background — takes 1–3 minutes for large manuals.
                    </p>
                  </form>
                )}
              </div>
            )}
          </div>

          {/* Feature chips */}
          <div className="mt-8 grid grid-cols-3 gap-3 text-center">
            {[
              { icon: "🎙️", label: "Hindi & English voice" },
              { icon: "📷", label: "Photo diagnosis" },
              { icon: "📖", label: "Manual-grounded only" },
            ].map((f) => (
              <div key={f.label} className="py-4 rounded-xl bg-white border border-slate-200 shadow-sm">
                <div className="text-2xl mb-1">{f.icon}</div>
                <p className="text-xs font-medium text-slate-600">{f.label}</p>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function BikeCard({ bike, onSelect }: { bike: BikeEntry; onSelect: (id: string) => void }) {
  return (
    <button
      onClick={() => onSelect(bike.document_id)}
      className="w-full flex items-center gap-4 p-4 rounded-xl border border-slate-200 hover:border-blue-400
                 hover:bg-blue-50/40 transition-all text-left group shadow-sm hover:shadow-md"
    >
      <div className="w-11 h-11 rounded-xl sarvam-gradient flex items-center justify-center shrink-0">
        <Bike size={20} className="text-white" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
          {bike.bike_brand} {bike.bike_model}
        </p>
        <p className="text-xs text-slate-400 mt-0.5">
          {bike.bike_year} · {bike.manual_type.replace(/_/g, " ")} · {bike.total_chunks.toLocaleString()} chunks
        </p>
      </div>
      <ChevronRight size={16} className="text-slate-300 group-hover:text-blue-500 group-hover:translate-x-0.5 transition-all shrink-0" />
    </button>
  );
}

function EmptyLibrary({ onUpload }: { onUpload: () => void }) {
  return (
    <div className="text-center py-12">
      <div className="w-14 h-14 rounded-2xl bg-slate-100 flex items-center justify-center mx-auto mb-4">
        <Library size={22} className="text-slate-400" />
      </div>
      <p className="font-medium text-slate-700 mb-1">No manuals indexed yet</p>
      <p className="text-sm text-slate-400 mb-5">Upload a PDF service manual to get started</p>
      <button
        onClick={onUpload}
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl sarvam-gradient text-white text-sm font-medium shadow"
      >
        <Upload size={14} /> Upload Manual
      </button>
    </div>
  );
}

function UploadSuccess({ brand, model, onStart }: { brand: string; model: string; onStart: () => void }) {
  return (
    <div className="text-center py-10">
      <div className="w-14 h-14 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-4">
        <CheckCircle2 size={24} className="text-green-600" />
      </div>
      <p className="font-semibold text-slate-900 text-lg mb-1">Your manual is ready!</p>
      <p className="text-sm text-slate-500 mb-6">
        <span className="font-medium text-slate-700">{brand} {model}</span> has been indexed.
        Every answer will come directly from this manual.
      </p>
      <button
        onClick={onStart}
        className="inline-flex items-center gap-2 px-6 py-3 rounded-xl sarvam-gradient text-white font-semibold shadow-md hover:opacity-90"
      >
        Ask your first question <ChevronRight size={16} />
      </button>
    </div>
  );
}

function LabeledInput({ label, placeholder, value, onChange }: {
  label: string; placeholder: string; value: string; onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-600 mb-1.5">{label}</label>
      <input
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2.5 text-sm rounded-lg border border-slate-200 bg-white
                   placeholder:text-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      />
    </div>
  );
}
