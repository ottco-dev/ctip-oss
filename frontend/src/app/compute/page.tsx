'use client';

/**
 * Compute network — every agent that lends GPU/CPU time to this CTIP coordinator, and the job queue.
 * Workers enrol with a one-time token (`ctip-worker enroll`); jobs run one per worker (parallel jobs),
 * DDP only inside one multi-GPU worker. Polls every 5 s.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Activity, Ban, Copy, Cpu, Gauge, Loader2, Plug, Server, Square, Timer } from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';

interface Capabilities {
  backend: 'cuda' | 'rocm' | 'mps' | 'cpu';
  device: string;
  gpus: number;
  vram_gb: number;
  ram_gb: number;
  cpu_cores: number;
  torch?: string;
  os?: string;
}

interface Worker {
  id: string;
  name: string;
  online: boolean;
  revoked: boolean;
  last_seen: number;
  created_at: number;
  capabilities: Partial<Capabilities>;
  free: { vram_gb?: number; ram_gb?: number; busy_by_others?: boolean };
}

interface Job {
  id: string;
  name: string;
  kind: string;
  status: 'queued' | 'leased' | 'running' | 'completed' | 'failed' | 'cancelled';
  attempts: number;
  max_attempts: number;
  worker_id: string | null;
  progress: number;
  epoch: number | null;
  metrics: Record<string, number>;
  message: string;
  spec: Record<string, unknown>;
  checkpoint: string | null;
  results: string[];
  cancel_requested: boolean;
  created_at: number;
  updated_at: number;
}

const BACKEND_LABEL: Record<string, string> = { cuda: 'CUDA', rocm: 'ROCm', mps: 'Apple MPS', cpu: 'CPU' };

const STATUS_STYLE: Record<Job['status'], string> = {
  queued: 'text-text-secondary border-border-muted',
  leased: 'text-accent-text border-accent/40',
  running: 'text-accent-text border-accent/40 bg-accent/10',
  completed: 'text-green-400 border-green-500/30',
  failed: 'text-red-400 border-red-500/30',
  cancelled: 'text-text-muted border-border',
};

function ago(t: number): string {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

function Bar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn('h-1.5 rounded-full overflow-hidden', className)} style={{ background: 'var(--border)' }}>
      <div className="h-full bg-accent transition-all" style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }} />
    </div>
  );
}

function Stat({ icon: Icon, label, value, hint }: { icon: React.ElementType; label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border px-4 py-3 min-w-0" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
        <Icon className="w-3.5 h-3.5" /> {label}
      </div>
      <div className="mt-1 text-xl font-semibold font-mono text-text-primary tabular-nums">{value}</div>
      {hint && <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{hint}</div>}
    </div>
  );
}

function ConnectPanel({ onClose }: { onClose: () => void }) {
  const [note, setNote] = useState('');
  const [token, setToken] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const create = useMutation({
    mutationFn: async () => (await api.post('/compute/enrollments', { note, ttl_hours: 24 })).data as { token: string },
    onSuccess: (d) => setToken(d.token),
  });
  const origin = typeof window !== 'undefined' ? window.location.origin : 'https://your-ctip';
  const cmd = token ? `ctip-worker enroll --server ${origin} --token ${token} --name "${note || 'my-pc'}"` : '';
  return (
    <div className="rounded-xl border p-4 space-y-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text-primary">Connect a worker</h3>
        <button onClick={onClose} className="text-xs" style={{ color: 'var(--text-muted)' }}>Close</button>
      </div>
      {!token ? (
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-1 text-xs flex-1 min-w-[200px]" style={{ color: 'var(--text-secondary)' }}>
            Who is it for? (shown in the list)
            <input id="enroll-note" value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. ottco RTX 4060"
              className="px-3 py-2 rounded-lg border text-sm text-text-primary" style={{ borderColor: 'var(--border)', background: 'var(--background)' }} />
          </label>
          <button onClick={() => create.mutate()} disabled={create.isPending}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium bg-accent hover:bg-accent-hover text-white disabled:opacity-50">
            {create.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plug className="w-4 h-4" />} Create one-time token
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            Valid for 24 hours, usable once. It is shown only now. On the worker machine (CTIP installed with <code>uv pip install -e .</code>), run:
          </p>
          <div className="flex items-start gap-2">
            <pre className="flex-1 min-w-0 overflow-x-auto text-xs font-mono p-3 rounded-lg text-text-primary" style={{ background: 'var(--background)', border: '1px solid var(--border)' }}>{cmd}</pre>
            <button onClick={() => navigator.clipboard?.writeText(cmd).then(() => setCopied(true)).catch(() => setCopied(false))}
              className="flex items-center gap-1 px-2.5 py-2 rounded-lg text-xs border" style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}>
              <Copy className="w-3.5 h-3.5" /> {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Then <code>ctip-worker run</code> — the machine appears below within seconds.</p>
        </div>
      )}
      {create.isError && <p className="text-xs text-red-400">Could not create a token: {(create.error as Error).message}</p>}
    </div>
  );
}

export default function ComputePage() {
  const qc = useQueryClient();
  const [connect, setConnect] = useState(false);
  const [confirmRevoke, setConfirmRevoke] = useState<string | null>(null);
  const [showRevoked, setShowRevoked] = useState(false);
  const workers = useQuery({ queryKey: ['compute', 'workers'], queryFn: async () => (await api.get('/compute/workers')).data as Worker[], refetchInterval: 5000 });
  const jobs = useQuery({ queryKey: ['compute', 'jobs'], queryFn: async () => (await api.get('/compute/jobs?limit=100')).data as Job[], refetchInterval: 5000 });
  const refresh = () => qc.invalidateQueries({ queryKey: ['compute'] });
  const cancel = useMutation({ mutationFn: (id: string) => api.post(`/compute/jobs/${id}/cancel`), onSuccess: refresh });
  const revoke = useMutation({ mutationFn: (id: string) => api.post(`/compute/workers/${id}/revoke`), onSuccess: refresh });
  const bench = useMutation({
    mutationFn: () => api.post('/compute/jobs', { name: 'Detection benchmark', spec: { kind: 'detection_benchmark', model: 'yolo11s', imgsz: 1280, runs: 20 }, requirements: { min_ram_gb: 1 } }),
    onSuccess: refresh,
  });

  const ws = useMemo(() => workers.data ?? [], [workers.data]);
  const js = jobs.data ?? [];
  const byWorker = useMemo(() => Object.fromEntries(ws.map((w) => [w.id, w])), [ws]);
  const online = ws.filter((w) => w.online);
  const revokedCount = ws.filter((w) => w.revoked).length;
  const shown = [...ws].filter((w) => showRevoked || !w.revoked)
    .sort((a, b) => Number(b.online) - Number(a.online) || Number(a.revoked) - Number(b.revoked) || b.last_seen - a.last_seen);
  const gpus = online.reduce((n, w) => n + (w.capabilities.backend && w.capabilities.backend !== 'cpu' ? w.capabilities.gpus ?? 0 : 0), 0);
  const vram = online.reduce((n, w) => n + (w.capabilities.vram_gb ?? 0) * Math.max(1, w.capabilities.gpus ?? 1), 0);
  const running = js.filter((j) => j.status === 'running' || j.status === 'leased');
  const queued = js.filter((j) => j.status === 'queued');

  return (
    <div className="p-6 space-y-6 max-w-[1400px]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-text-primary flex items-center gap-2"><Server className="w-5 h-5 text-accent-text" /> Compute network</h1>
          <p className="text-sm mt-1 max-w-2xl" style={{ color: 'var(--text-secondary)' }}>
            Machines that lend GPU or CPU time to this CTIP instance. Each job runs on one machine; jobs resume from
            their last checkpoint when a machine drops out or its owner needs the GPU back.
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => bench.mutate()} disabled={bench.isPending}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border disabled:opacity-50" style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}>
            <Gauge className="w-4 h-4" /> Queue benchmark
          </button>
          <button onClick={() => setConnect((v) => !v)} className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium bg-accent hover:bg-accent-hover text-white">
            <Plug className="w-4 h-4" /> Connect a worker
          </button>
        </div>
      </div>

      {connect && <ConnectPanel onClose={() => setConnect(false)} />}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat icon={Server} label="Workers online" value={`${online.length} / ${ws.filter((w) => !w.revoked).length}`} />
        <Stat icon={Cpu} label="GPUs online" value={String(gpus)} hint={`${vram.toFixed(0)} GB VRAM in total`} />
        <Stat icon={Activity} label="Running" value={String(running.length)} />
        <Stat icon={Timer} label="Queued" value={String(queued.length)} />
      </div>

      <section className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Workers</h2>
          {revokedCount > 0 && (
            <button onClick={() => setShowRevoked((v) => !v)} className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {showRevoked ? 'Hide' : 'Show'} revoked ({revokedCount})
            </button>
          )}
        </div>
        {workers.isLoading ? (
          <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}><Loader2 className="w-4 h-4 animate-spin" /> Loading…</div>
        ) : shown.length === 0 ? (
          <div className="rounded-xl border p-6 text-sm" style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)', background: 'var(--surface)' }}>
            No machine connected yet. Use <b>Connect a worker</b> to create a one-time token, then run <code>ctip-worker enroll</code> and <code>ctip-worker run</code> on the machine.
          </div>
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {shown.map((w) => {
              const c = w.capabilities;
              const job = js.find((j) => j.worker_id === w.id && (j.status === 'running' || j.status === 'leased'));
              const state = w.revoked ? 'revoked' : !w.online ? 'offline' : w.free.busy_by_others ? 'owner busy' : job ? 'working' : 'idle';
              const dot = w.revoked || !w.online ? 'var(--text-muted)' : w.free.busy_by_others ? '#d97706' : 'var(--accent)';
              return (
                <div key={w.id} className="rounded-xl border p-4 space-y-3 min-w-0" style={{ borderColor: 'var(--border)', background: 'var(--surface)', opacity: w.revoked ? 0.6 : 1 }}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full flex-none" style={{ background: dot }} />
                        <span className="font-semibold text-text-primary truncate">{w.name}</span>
                      </div>
                      <div className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-muted)' }}>{c.device ?? 'unknown device'}</div>
                    </div>
                    <span className="text-[10px] px-1.5 py-0.5 rounded border font-medium text-accent-text border-accent/30 flex-none">
                      {BACKEND_LABEL[c.backend ?? ''] ?? '?'}{(c.gpus ?? 0) > 1 ? ` ×${c.gpus}` : ''}
                    </span>
                  </div>
                  <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono tabular-nums">
                    <dt style={{ color: 'var(--text-muted)' }}>VRAM</dt><dd className="text-text-primary">{c.vram_gb ? `${(w.free.vram_gb ?? 0).toFixed(1)} / ${c.vram_gb} GB free` : '–'}</dd>
                    <dt style={{ color: 'var(--text-muted)' }}>RAM</dt><dd className="text-text-primary">{(w.free.ram_gb ?? 0).toFixed(1)} / {c.ram_gb ?? '?'} GB free</dd>
                    <dt style={{ color: 'var(--text-muted)' }}>CPU</dt><dd className="text-text-primary">{c.cpu_cores ?? '?'} threads</dd>
                    <dt style={{ color: 'var(--text-muted)' }}>State</dt><dd className="text-text-primary">{state} · {ago(w.last_seen)}</dd>
                  </dl>
                  {job && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs"><span className="text-text-primary truncate">{job.name || job.kind}</span><span className="font-mono" style={{ color: 'var(--text-muted)' }}>{Math.round(job.progress * 100)}%</span></div>
                      <Bar value={job.progress} />
                    </div>
                  )}
                  <div className="flex justify-between items-center text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    <span className="truncate">{[c.os, c.torch && `torch ${c.torch}`].filter(Boolean).join(' · ')}</span>
                    {!w.revoked && (confirmRevoke === w.id ? (
                      <span className="flex items-center gap-2">
                        <button onClick={() => { revoke.mutate(w.id); setConfirmRevoke(null); }} className="text-red-400 font-medium">Revoke for good</button>
                        <button onClick={() => setConfirmRevoke(null)}>Keep</button>
                      </span>
                    ) : (
                      <button onClick={() => setConfirmRevoke(w.id)} className="flex items-center gap-1 hover:text-red-400" title="Disconnect this machine; its token stops working">
                        <Ban className="w-3 h-3" /> Revoke
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Jobs</h2>
        <div className="rounded-xl border overflow-x-auto" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
                <th className="px-4 py-2 font-medium">Job</th><th className="px-4 py-2 font-medium">Status</th><th className="px-4 py-2 font-medium">Worker</th>
                <th className="px-4 py-2 font-medium w-48">Progress</th><th className="px-4 py-2 font-medium">Metrics</th><th className="px-4 py-2 font-medium">Updated</th><th />
              </tr>
            </thead>
            <tbody>
              {js.length === 0 && (
                <tr><td colSpan={7} className="px-4 py-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>No jobs yet — queue a benchmark to try a worker.</td></tr>
              )}
              {js.map((j) => {
                const m = Object.entries(j.metrics).slice(0, 3);
                return (
                  <tr key={j.id} style={{ borderBottom: '1px solid var(--border)' }} className="align-top">
                    <td className="px-4 py-2.5 min-w-[180px]">
                      <div className="text-text-primary font-medium">{j.name || j.kind}</div>
                      <div className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>{j.kind} · {j.id.slice(0, 8)} · try {j.attempts}/{j.max_attempts}{j.checkpoint ? ' · checkpoint' : ''}</div>
                      {j.message && <div className="text-[11px] mt-0.5 max-w-md truncate" title={j.message} style={{ color: 'var(--text-muted)' }}>{j.message.split('\n')[0]}</div>}
                    </td>
                    <td className="px-4 py-2.5"><span className={cn('text-[11px] px-1.5 py-0.5 rounded border capitalize', STATUS_STYLE[j.status])}>{j.cancel_requested && j.status !== 'cancelled' ? 'cancelling' : j.status}</span></td>
                    <td className="px-4 py-2.5 text-xs text-text-primary">{j.worker_id ? byWorker[j.worker_id]?.name ?? j.worker_id.slice(0, 8) : '–'}</td>
                    <td className="px-4 py-2.5"><Bar value={j.progress} /><div className="text-[11px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>{Math.round(j.progress * 100)}%{j.epoch ? ` · epoch ${j.epoch}` : ''}</div></td>
                    <td className="px-4 py-2.5 text-[11px] font-mono text-text-primary whitespace-nowrap">{m.length ? m.map(([k, v]) => `${k} ${v.toFixed(v < 10 ? 3 : 1)}`).join(' · ') : '–'}</td>
                    <td className="px-4 py-2.5 text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{ago(j.updated_at)}</td>
                    <td className="px-4 py-2.5 text-right">
                      {(j.status === 'queued' || j.status === 'running' || j.status === 'leased') && !j.cancel_requested && (
                        <button onClick={() => cancel.mutate(j.id)} className="flex items-center gap-1 text-xs hover:text-red-400" style={{ color: 'var(--text-muted)' }}><Square className="w-3 h-3" /> Cancel</button>
                      )}
                      {j.status === 'completed' && j.results.length > 0 && (
                        <a href={`/api/v1/compute/artifacts/${j.results[0]}`} className="text-xs text-accent-text hover:underline">Result</a>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
