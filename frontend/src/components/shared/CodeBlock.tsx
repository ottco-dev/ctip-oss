"use client";

import React, { useState } from "react";
import { Copy, Check } from "lucide-react";

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  maxHeight?: number;
  className?: string;
}

/**
 * CodeBlock — syntax-highlighted (via CSS classes) config/JSON viewer.
 * Uses simple text rendering — no heavy Prism/highlight.js dependency.
 */
export function CodeBlock({
  code,
  language = "yaml",
  title,
  maxHeight = 400,
  className = "",
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard not available
    }
  };

  return (
    <div
      className={`rounded-xl overflow-hidden ${className}`}
      style={{ border: "1px solid #21262d", background: "#0d1117" }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2"
        style={{ background: "#161b22", borderBottom: "1px solid #21262d" }}
      >
        <div className="flex items-center gap-2">
          {title && (
            <span className="text-xs font-medium" style={{ color: "#e6edf3" }}>
              {title}
            </span>
          )}
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{ background: "#21262d", color: "#8b949e" }}
          >
            {language}
          </span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-[11px] px-2 py-1 rounded transition-colors"
          style={{
            color: copied ? "#4ade80" : "#8b949e",
            background: "transparent",
          }}
        >
          {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      {/* Code */}
      <pre
        className="overflow-auto p-4 text-[12px] font-mono leading-relaxed"
        style={{
          maxHeight: `${maxHeight}px`,
          color: "#e6edf3",
          whiteSpace: "pre-wrap",
          wordBreak: "break-all",
        }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}
