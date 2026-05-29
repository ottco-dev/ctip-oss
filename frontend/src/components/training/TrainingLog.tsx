'use client';

import { useEffect, useRef, useState } from 'react';
import { Terminal, Trash2, ArrowDownToLine, Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useTrainingStore } from '@/store/trainingStore';
import type { LogLevel } from '@/lib/types';

const LEVEL_STYLES: Record<LogLevel, string> = {
  info:    'text-text-secondary',
  success: 'text-status-success',
  warning: 'text-status-warning',
  error:   'text-status-error',
  dim:     'text-text-muted/60',
  header:  'text-accent font-semibold tracking-wide',
};

const LEVEL_BADGE: Partial<Record<LogLevel, string>> = {
  warning: 'text-status-warning',
  error:   'text-status-error',
  success: 'text-status-success',
};

interface TrainingLogProps {
  wsConnected: boolean;
  className?: string;
}

export function TrainingLog({ wsConnected, className }: TrainingLogProps) {
  const { logLines, clearLog } = useTrainingStore();
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new lines arrive
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [logLines, autoScroll]);

  // Detect manual scroll up to pause auto-scroll
  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    setAutoScroll(atBottom);
  };

  return (
    <div className={cn('card flex flex-col', className)}>
      {/* Header */}
      <div className="card-header flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-text-secondary" />
          <span>Training Log</span>
          {logLines.length > 0 && (
            <span className="text-xs text-text-muted font-mono">({logLines.length} lines)</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* WS connection indicator */}
          <div className={cn('flex items-center gap-1 text-xs', wsConnected ? 'text-status-success' : 'text-text-muted')}>
            {wsConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            <span>{wsConnected ? 'live' : 'offline'}</span>
          </div>

          {/* Auto-scroll toggle */}
          <button
            onClick={() => setAutoScroll((v) => !v)}
            className={cn(
              'flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors',
              autoScroll
                ? 'bg-accent/20 text-accent'
                : 'text-text-muted hover:text-text-secondary',
            )}
            title={autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'}
          >
            <ArrowDownToLine className="w-3 h-3" />
          </button>

          {/* Clear */}
          <button
            onClick={clearLog}
            className="text-text-muted hover:text-text-secondary transition-colors p-1 rounded"
            title="Clear log"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Log body */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto bg-[#0d0d0d] rounded-b-lg font-mono text-[11px] leading-[1.6] p-3 min-h-[200px] max-h-[340px]"
      >
        {logLines.length === 0 ? (
          <div className="text-text-muted/40 select-none">
            {wsConnected
              ? 'Waiting for training to start…'
              : 'Connect WebSocket to stream training output.'}
          </div>
        ) : (
          <>
            {logLines.map((entry, i) => (
              <div key={i} className="flex gap-2 hover:bg-white/[0.02] px-1 rounded -mx-1 group">
                {/* Timestamp */}
                <span className="text-text-muted/30 flex-shrink-0 w-[52px] text-right select-none">
                  {new Date(entry.ts * 1000).toLocaleTimeString('de-DE', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  })}
                </span>

                {/* Level badge (only for warning/error/success) */}
                {LEVEL_BADGE[entry.level] && (
                  <span className={cn('flex-shrink-0 uppercase text-[9px] font-bold w-[36px]', LEVEL_BADGE[entry.level])}>
                    {entry.level.slice(0, 4)}
                  </span>
                )}
                {!LEVEL_BADGE[entry.level] && <span className="w-[36px] flex-shrink-0" />}

                {/* Line content */}
                <span className={cn('flex-1 break-all whitespace-pre-wrap', LEVEL_STYLES[entry.level])}>
                  {entry.line}
                </span>
              </div>
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>
    </div>
  );
}
