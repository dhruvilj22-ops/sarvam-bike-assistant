"use client";
import { ChevronRight } from "lucide-react";

export default function FollowupChips({
  followups,
  onSelect,
}: {
  followups: string[];
  onSelect: (q: string) => void;
}) {
  if (!followups?.length) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {followups.map((q, i) => (
        <button
          key={i}
          onClick={() => onSelect(q)}
          className="flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium
                     bg-white border border-slate-200 text-slate-600 hover:border-blue-400
                     hover:text-blue-600 hover:bg-blue-50 transition-all shadow-sm"
        >
          {q}
          <ChevronRight size={11} />
        </button>
      ))}
    </div>
  );
}
