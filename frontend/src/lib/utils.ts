/**
 * Utility functions for the frontend.
 */

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { MaturityStage, RunStatus, JobStatus } from './types';

/** Merge Tailwind classes safely. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a number as a percentage. */
export function pct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format a duration in seconds to human-readable. */
export function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

/** Format a Unix timestamp to relative time. */
export function timeAgo(timestamp: number): string {
  const seconds = Math.floor(Date.now() / 1000 - timestamp);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** Format bytes to human-readable. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/** Format VRAM GB value. */
export function formatVram(gb: number): string {
  return `${gb.toFixed(1)} GB`;
}

/** Format a Date to relative human-readable string (e.g. "2 hours ago"). */
export function formatDistanceToNow(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

// ── MATURITY COLOR MAPPING ─────────────────────────────────────

const MATURITY_COLORS: Record<MaturityStage | string, string> = {
  clear: '#60a5fa',
  cloudy: '#f9fafb',
  amber: '#f59e0b',
  cloudy_amber_mix: '#d97706',
  degraded: '#6b7280',
  unknown: '#4b5563',
};

const MATURITY_LABELS: Record<MaturityStage | string, string> = {
  clear: 'Clear',
  cloudy: 'Cloudy',
  amber: 'Amber',
  cloudy_amber_mix: 'Cloudy/Amber Mix',
  degraded: 'Degraded',
  unknown: 'Unknown',
};

export function getMaturityColor(stage: string): string {
  return MATURITY_COLORS[stage] ?? '#4b5563';
}

export function getMaturityLabel(stage: string): string {
  return MATURITY_LABELS[stage] ?? stage;
}

// ── STATUS COLORS ──────────────────────────────────────────────

export function getStatusColor(status: RunStatus | JobStatus): string {
  const colors: Record<string, string> = {
    pending: '#8b949e',
    running: '#3b82f6',
    completed: '#22c55e',
    failed: '#ef4444',
    stopped: '#eab308',
    cancelled: '#6b7280',
  };
  return colors[status] ?? '#8b949e';
}

export function getStatusBadgeClass(status: RunStatus | JobStatus): string {
  const classes: Record<string, string> = {
    pending: 'badge-muted',
    running: 'badge-info',
    completed: 'badge-success',
    failed: 'badge-error',
    stopped: 'badge-warning',
    cancelled: 'badge-muted',
  };
  return classes[status] ?? 'badge-muted';
}

// ── TRICHOME TYPE COLORS ───────────────────────────────────────

const TRICHOME_TYPE_COLORS: Record<string, string> = {
  capitate_stalked: '#22d3ee',
  capitate_sessile: '#34d399',
  bulbous: '#a78bfa',
  non_glandular: '#fb923c',
  unknown: '#4b5563',
};

export function getTrichomeTypeColor(type: string): string {
  return TRICHOME_TYPE_COLORS[type] ?? '#4b5563';
}

// ── CONFIDENCE FORMATTING ──────────────────────────────────────

export function formatConfidence(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export function getConfidenceColor(value: number): string {
  if (value >= 0.75) return '#22c55e'; // green
  if (value >= 0.50) return '#eab308'; // yellow
  return '#ef4444'; // red
}

// ── NUMBER FORMATTING ──────────────────────────────────────────

export function formatNumber(n: number, decimals = 0): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(decimals);
}
