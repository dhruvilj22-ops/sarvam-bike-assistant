"use client";

const CONFIGS: Record<string, { label: string; cls: string }> = {
  Urgent: {
    label: "Urgent",
    cls: "bg-red-100 text-red-700 border border-red-200",
  },
  "Get Checked Soon": {
    label: "Get Checked Soon",
    cls: "bg-orange-100 text-orange-700 border border-orange-200",
  },
  "Monitor It": {
    label: "Monitor It",
    cls: "bg-yellow-100 text-yellow-700 border border-yellow-200",
  },
  Informational: {
    label: "Informational",
    cls: "bg-blue-100 text-blue-700 border border-blue-200",
  },
};

export default function SeverityBadge({ label }: { label: string }) {
  const cfg = CONFIGS[label] ?? {
    label,
    cls: "bg-slate-100 text-slate-600 border border-slate-200",
  };
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${cfg.cls}`}>
      {cfg.label}
    </span>
  );
}
