"use client";
import { BookOpen } from "lucide-react";
import type { Citation } from "@/lib/api";

export default function CitationBlock({ citations }: { citations: Citation[] }) {
  if (!citations?.length) return null;
  return (
    <div className="mt-3 space-y-1.5">
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Sources</p>
      {citations.map((c, i) => (
        <div
          key={i}
          className="flex items-start gap-2 px-3 py-2 rounded-lg bg-blue-50 border border-blue-100"
        >
          <BookOpen size={13} className="mt-0.5 shrink-0 text-blue-500" />
          <div>
            <span className="text-xs font-semibold text-blue-700">
              {c.section_number && `§${c.section_number} — `}{c.section_title}
            </span>
            {c.page_number > 0 && (
              <span className="ml-1.5 text-xs text-blue-400">p. {c.page_number}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
