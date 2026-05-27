'use client';

/**
 * CTIP Full Installer & Setup Wizard
 *
 * 9-step OS-style first-run assistant:
 *   0  Welcome
 *   1  System Check      — live dependency + service check with log
 *   2  Network           — domain / nginx port
 *   3  Hardware          — CUDA / VRAM
 *   4  Storage           — data directories
 *   5  Label Studio      — full LS setup: test + create project
 *   6  Services          — MLflow, W&B
 *   7  Security          — secret key, API token
 *   8  Review            — summary
 *   9  Verification      — post-save health check with live log
 */

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Microscope, Globe, Cpu, HardDrive, Plug, ShieldCheck,
  CheckCircle2, ChevronRight, ChevronLeft, Loader2,
  AlertTriangle, Info, Eye, EyeOff, ExternalLink,
  RotateCcw, Terminal, RefreshCw, Tags, Activity,
  XCircle, CheckCircle, Clock,
} from 'lucide-react';
import { api } from '@/lib/api';

// ── Types ─────────────────────────────────────────────────────────────────────

interface WizardState {
  publicDomain: string; publicPort: string;
  cudaDevice: string; cudaVisible: string; vramLimit: string; vramInference: string;
  dataRoot: string; modelsDir: string; outputsDir: string;
  labelStudioUrl: string; labelStudioKey: string;
  labelStudioProjectName: string; labelStudioProjectId: number;
  mlflowUri: string; mlflowExperiment: string;
  useWandb: boolean; wandbKey: string; wandbProject: string;
  secretKey: string; apiToken: string; environment: string;
}

type WizardAction =
  | { type: 'SET'; field: keyof WizardState; value: string | boolean | number }
  | { type: 'LOAD'; payload: Partial<WizardState> };

interface CheckItem {
  name: string; ok: boolean; value: string; detail: string; required: boolean;
}
interface VerificationItem {
  name: string; url: string; ok: boolean;
  status_code: number; latency_ms: number; detail: string;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const STEPS = [
  { id: 'welcome',      label: 'Welcome',       icon: Microscope },
  { id: 'syscheck',     label: 'System Check',  icon: Activity },
  { id: 'network',      label: 'Network',        icon: Globe },
  { id: 'hardware',     label: 'Hardware',       icon: Cpu },
  { id: 'storage',      label: 'Storage',        icon: HardDrive },
  { id: 'labelstudio',  label: 'Label Studio',   icon: Tags },
  { id: 'services',     label: 'Services',       icon: Plug },
  { id: 'security',     label: 'Security',       icon: ShieldCheck },
  { id: 'review',       label: 'Review',         icon: CheckCircle2 },
] as const;

type StepId = (typeof STEPS)[number]['id'];

const DEFAULT_STATE: WizardState = {
  publicDomain: '', publicPort: '3001',
  cudaDevice: 'cuda:0', cudaVisible: '0', vramLimit: '8.0', vramInference: '2.0',
  dataRoot: '/path/to/trichome-analysis/data',
  modelsDir: '/path/to/trichome-analysis/data/models',
  outputsDir: '/path/to/trichome-analysis/data/outputs',
  labelStudioUrl: 'http://localhost:3005', labelStudioKey: '',
  labelStudioProjectName: 'CTIP — Trichome Detection', labelStudioProjectId: 0,
  mlflowUri: 'http://localhost:3004', mlflowExperiment: 'trichome-detection',
  useWandb: false, wandbKey: '', wandbProject: 'trichome-detection',
  secretKey: '', apiToken: '', environment: 'development',
};

const SESSION_KEY = 'ctip-setup-draft';

// ── Reducer ───────────────────────────────────────────────────────────────────

function reducer(state: WizardState, action: WizardAction): WizardState {
  if (action.type === 'SET') return { ...state, [action.field]: action.value };
  if (action.type === 'LOAD') return { ...state, ...action.payload };
  return state;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function generateSecretKey() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
  return Array.from({ length: 64 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

function stateToEnvMap(s: WizardState): Record<string, string> {
  return {
    PUBLIC_DOMAIN: s.publicDomain, PUBLIC_PORT: s.publicPort,
    CUDA_DEVICE: s.cudaDevice, CUDA_VISIBLE_DEVICES: s.cudaVisible,
    VRAM_LIMIT_GB: s.vramLimit, VRAM_INFERENCE_BUDGET_GB: s.vramInference,
    DATA_ROOT: s.dataRoot, MODELS_DIR: s.modelsDir, OUTPUTS_DIR: s.outputsDir,
    LABEL_STUDIO_URL: s.labelStudioUrl, LABEL_STUDIO_API_KEY: s.labelStudioKey,
    MLFLOW_TRACKING_URI: s.mlflowUri, MLFLOW_EXPERIMENT_NAME: s.mlflowExperiment,
    USE_WANDB: s.useWandb ? 'true' : 'false',
    WANDB_API_KEY: s.wandbKey, WANDB_PROJECT: s.wandbProject,
    SECRET_KEY: s.secretKey, API_TOKEN: s.apiToken, ENVIRONMENT: s.environment,
  };
}

function validateStep(step: StepId, s: WizardState): Record<string, string> {
  const e: Record<string, string> = {};
  if (step === 'network') {
    if (s.publicDomain && !/^[a-zA-Z0-9]([a-zA-Z0-9\-.]{0,253}[a-zA-Z0-9])?$/.test(s.publicDomain))
      e.publicDomain = 'Invalid domain format';
    if (isNaN(+s.publicPort) || +s.publicPort < 1 || +s.publicPort > 65535)
      e.publicPort = 'Port must be 1–65535';
  }
  if (step === 'hardware') {
    if (isNaN(+s.vramLimit) || +s.vramLimit < 1 || +s.vramLimit > 80)
      e.vramLimit = 'Must be 1–80 GB';
  }
  if (step === 'storage') {
    if (!s.dataRoot.startsWith('/')) e.dataRoot = 'Must be an absolute path';
  }
  if (step === 'services') {
    if (s.mlflowUri && !/^https?:\/\//.test(s.mlflowUri)) e.mlflowUri = 'Must be HTTP/HTTPS';
  }
  return e;
}

// ── Shared UI primitives ──────────────────────────────────────────────────────

function InputField({ label, value, onChange, placeholder, hint, error, type = 'text', monospace = false, action }: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; hint?: string; error?: string;
  type?: string; monospace?: boolean;
  action?: { label: string; onClick: () => void };
}) {
  const [show, setShow] = useState(false);
  const isPw = type === 'password';
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-text-secondary uppercase tracking-wider">{label}</label>
        {action && (
          <button type="button" onClick={action.onClick}
            className="text-xs text-accent hover:underline">{action.label}</button>
        )}
      </div>
      <div className="relative">
        <input type={isPw && !show ? 'password' : 'text'} value={value}
          onChange={e => onChange(e.target.value)} placeholder={placeholder} spellCheck={false}
          className={['input', monospace ? 'font-mono text-xs' : '', isPw ? 'pr-10' : '',
            error ? 'border-red-500 focus:border-red-500 focus:shadow-[0_0_0_1px_#ef4444]' : ''].filter(Boolean).join(' ')} />
        {isPw && (
          <button type="button" onClick={() => setShow(p => !p)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary">
            {show ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        )}
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {hint && !error && <p className="text-xs text-text-muted">{hint}</p>}
    </div>
  );
}

function Toggle({ label, value, onChange, hint }: {
  label: string; value: boolean; onChange: (v: boolean) => void; hint?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <div>
        <p className="text-sm text-text-primary">{label}</p>
        {hint && <p className="text-xs text-text-muted mt-0.5">{hint}</p>}
      </div>
      <button type="button" onClick={() => onChange(!value)} role="switch" aria-checked={value}
        className={['relative inline-flex h-6 w-11 shrink-0 rounded-full border-2 border-transparent transition-colors duration-200',
          value ? 'bg-accent' : 'bg-border'].join(' ')}>
        <span className={['pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform transition-transform duration-200',
          value ? 'translate-x-5' : 'translate-x-0'].join(' ')} />
      </button>
    </div>
  );
}

function InfoBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-md p-3 text-xs"
      style={{ background: 'rgba(59,130,246,0.07)', border: '1px solid rgba(59,130,246,0.2)', color: '#60a5fa' }}>
      <Info className="w-4 h-4 mt-0.5 shrink-0" /><span>{children}</span>
    </div>
  );
}

function WarnBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-md p-3 text-xs"
      style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.2)', color: '#eab308' }}>
      <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" /><span>{children}</span>
    </div>
  );
}

function StatusDot({ ok, pending }: { ok?: boolean; pending?: boolean }) {
  if (pending) return <Loader2 className="w-3.5 h-3.5 text-text-muted animate-spin shrink-0" />;
  if (ok) return <CheckCircle className="w-3.5 h-3.5 text-status-success shrink-0" />;
  return <XCircle className="w-3.5 h-3.5 text-status-error shrink-0" />;
}

// ── Step: Welcome ─────────────────────────────────────────────────────────────

function StepWelcome() {
  return (
    <div className="text-center space-y-6">
      <div className="flex justify-center">
        <div className="w-24 h-24 rounded-2xl flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg,rgba(35,134,54,.2),rgba(59,130,246,.15))', border: '1px solid rgba(35,134,54,.4)' }}>
          <Microscope className="w-12 h-12 text-accent" />
        </div>
      </div>
      <div>
        <h1 className="text-3xl font-bold text-text-primary mb-1">Welcome to CTIP</h1>
        <p className="text-text-muted text-sm">Cannabis Trichome Intelligence Platform — Full Installer</p>
      </div>
      <p className="text-text-secondary text-sm leading-relaxed">
        This installer checks your system, configures all services, sets up Label Studio
        with a ready-to-use annotation project, and verifies every subsystem before you start.
      </p>
      <div className="grid grid-cols-3 gap-3 text-left">
        {[
          { icon: Activity, label: 'System Check', desc: 'All dependencies' },
          { icon: Tags,     label: 'Label Studio', desc: 'Full LS setup' },
          { icon: CheckCircle2, label: 'Verification', desc: 'Live health check' },
        ].map(({ icon: Icon, label, desc }) => (
          <div key={label} className="flex flex-col gap-2 rounded-lg p-3"
            style={{ background: '#161b22', border: '1px solid #21262d' }}>
            <Icon className="w-4 h-4 text-accent" />
            <p className="text-xs font-medium text-text-primary">{label}</p>
            <p className="text-xs text-text-muted">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Step: System Check ────────────────────────────────────────────────────────

function StepSystemCheck() {
  const [items, setItems] = useState<CheckItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [ran, setRan] = useState(false);
  const [allOk, setAllOk] = useState<boolean | null>(null);

  const run = async () => {
    setLoading(true);
    setItems([]);
    try {
      const res = await api.get('/setup/system-check');
      setItems(res.data.items);
      setAllOk(res.data.all_required_ok);
      setRan(true);
    } catch {
      setItems([{ name: 'API', ok: false, value: '', detail: 'Backend not reachable', required: true }]);
      setAllOk(false);
      setRan(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { run(); }, []);

  const core = items.filter(i => !i.name.startsWith('pkg:'));
  const pkgs = items.filter(i => i.name.startsWith('pkg:'));

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-text-primary">System Check</h2>
          <p className="text-sm text-text-muted mt-1">Verifying all required dependencies and services.</p>
        </div>
        <button onClick={run} disabled={loading}
          className="btn-secondary text-xs flex items-center gap-1.5">
          <RefreshCw className={['w-3.5 h-3.5', loading ? 'animate-spin' : ''].join(' ')} />
          Re-run
        </button>
      </div>

      {loading && items.length === 0 && (
        <div className="flex items-center gap-3 py-8 justify-center text-text-muted">
          <Loader2 className="w-5 h-5 animate-spin text-accent" />
          <span className="text-sm">Checking system…</span>
        </div>
      )}

      {core.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Environment</p>
          <div className="rounded-lg overflow-hidden" style={{ border: '1px solid #21262d' }}>
            {core.map(item => (
              <div key={item.name}
                className="flex items-center gap-3 px-3 py-2 border-b border-border last:border-0 text-sm"
                style={{ background: !item.ok && item.required ? 'rgba(239,68,68,0.04)' : undefined }}>
                <StatusDot ok={item.ok} />
                <span className={item.required ? 'text-text-primary' : 'text-text-secondary'}>{item.name}</span>
                {!item.required && <span className="text-[10px] text-text-muted border border-border rounded px-1">optional</span>}
                <span className="ml-auto font-mono text-xs text-text-muted truncate max-w-[160px]">
                  {item.ok ? item.value : item.detail}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {pkgs.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Python Packages</p>
          <div className="grid grid-cols-2 gap-1.5">
            {pkgs.map(item => (
              <div key={item.name}
                className="flex items-center gap-2 rounded px-2.5 py-1.5 text-xs"
                style={{ background: '#0d1117', border: `1px solid ${item.ok ? '#21262d' : item.required ? 'rgba(239,68,68,0.3)' : '#21262d'}` }}>
                <StatusDot ok={item.ok} />
                <span className="text-text-secondary font-mono">{item.name.replace('pkg:', '')}</span>
                <span className="ml-auto text-text-muted truncate">{item.ok ? item.value : 'missing'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {ran && allOk === false && (
        <WarnBox>
          Some required dependencies are missing. You can continue setup, but the platform may not work correctly.
          Install missing packages via <code>uv pip install -e &quot;.[all]&quot;</code>.
        </WarnBox>
      )}
      {ran && allOk === true && (
        <div className="flex items-center gap-2 rounded-md p-3 text-xs"
          style={{ background: 'rgba(34,197,94,0.07)', border: '1px solid rgba(34,197,94,0.2)', color: '#22c55e' }}>
          <CheckCircle className="w-4 h-4 shrink-0" />
          All required dependencies present — system ready.
        </div>
      )}
    </div>
  );
}

// ── Step: Network ─────────────────────────────────────────────────────────────

function StepNetwork({ state, set, errors }: {
  state: WizardState;
  set: (f: keyof WizardState) => (v: string | boolean) => void;
  errors: Record<string, string>;
}) {
  const isPublic = state.publicDomain.trim().length > 0;
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Network Access</h2>
        <p className="text-sm text-text-muted mt-1">Default: localhost only. Set a domain to expose publicly via nginx.</p>
      </div>
      <InfoBox>By default nginx only binds to localhost:3001 — not reachable from outside. Enable public access by entering your domain below.</InfoBox>
      <Toggle label="Enable public access" hint="Expose via nginx to your DDNS domain"
        value={isPublic} onChange={v => { if (!v) (set('publicDomain') as (v: string) => void)(''); }} />
      {isPublic && (
        <InputField label="Public Domain" value={state.publicDomain}
          onChange={set('publicDomain') as (v: string) => void}
          placeholder="mylab.ddns.net" hint="Hostname only — no http:// prefix" error={errors.publicDomain} />
      )}
      <InputField label="Public Port" value={state.publicPort}
        onChange={set('publicPort') as (v: string) => void}
        placeholder="3001" hint="nginx listens on this host port" error={errors.publicPort} />
      {!isPublic && (
        <div className="rounded-md p-3 text-xs space-y-1" style={{ background: '#0d1117', border: '1px solid #21262d' }}>
          <p className="font-medium text-text-secondary">Access URLs (localhost only)</p>
          <p className="text-text-muted">Frontend: <span className="text-accent">http://localhost:{state.publicPort}</span></p>
          <p className="text-text-muted">API: <span className="text-accent">http://localhost:8000/api/v1</span></p>
        </div>
      )}
    </div>
  );
}

// ── Step: Hardware ────────────────────────────────────────────────────────────

function StepHardware({ state, set, errors }: {
  state: WizardState;
  set: (f: keyof WizardState) => (v: string | boolean) => void;
  errors: Record<string, string>;
}) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">GPU & Hardware</h2>
        <p className="text-sm text-text-muted mt-1">One GPU task runs at a time — optimized for RTX 4060 (8 GB).</p>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <InputField label="CUDA Device" value={state.cudaDevice}
          onChange={set('cudaDevice') as (v: string) => void} placeholder="cuda:0" />
        <InputField label="CUDA_VISIBLE_DEVICES" value={state.cudaVisible}
          onChange={set('cudaVisible') as (v: string) => void} placeholder="0" />
        <InputField label="Total VRAM (GB)" value={state.vramLimit}
          onChange={set('vramLimit') as (v: string) => void} placeholder="8.0" error={errors.vramLimit} />
        <InputField label="Inference Reserve (GB)" value={state.vramInference}
          onChange={set('vramInference') as (v: string) => void} placeholder="2.0" />
      </div>
      <div>
        <label className="block text-xs font-medium text-text-secondary uppercase tracking-wider mb-1.5">Environment</label>
        <select value={state.environment}
          onChange={e => (set('environment') as (v: string) => void)(e.target.value)}
          className="input">
          <option value="development">Development</option>
          <option value="production">Production</option>
        </select>
      </div>
      <div className="rounded-lg p-3 space-y-2" style={{ background: '#0d1117', border: '1px solid #21262d' }}>
        <p className="text-xs text-text-secondary font-medium">VRAM Budget</p>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 rounded-full" style={{ background: '#21262d' }}>
            <div className="h-2 rounded-full bg-accent transition-all"
              style={{ width: `${Math.min(100, (+state.vramInference / +state.vramLimit) * 100)}%` }} />
          </div>
          <span className="text-xs text-text-muted whitespace-nowrap">
            {state.vramInference} / {state.vramLimit} GB inference reserve
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Step: Storage ─────────────────────────────────────────────────────────────

function StepStorage({ state, set, errors }: {
  state: WizardState;
  set: (f: keyof WizardState) => (v: string | boolean) => void;
  errors: Record<string, string>;
}) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Storage Paths</h2>
        <p className="text-sm text-text-muted mt-1">Absolute paths. Directories are created automatically on startup.</p>
      </div>
      <InfoBox>Tilde (~) is expanded. Datasets are tracked via DVC — these directories hold runtime data, not raw annotated sets.</InfoBox>
      <InputField label="Data Root" value={state.dataRoot}
        onChange={set('dataRoot') as (v: string) => void}
        placeholder="/path/to/trichome-analysis/data" monospace error={errors.dataRoot}
        hint="Parent for all data subdirectories" />
      <InputField label="Models Directory" value={state.modelsDir}
        onChange={set('modelsDir') as (v: string) => void}
        placeholder="/path/to/trichome-analysis/data/models" monospace hint="Trained weight files (.pt, .engine)" />
      <InputField label="Outputs Directory" value={state.outputsDir}
        onChange={set('outputsDir') as (v: string) => void}
        placeholder="/path/to/trichome-analysis/data/outputs" monospace hint="Detection results, PDFs, CSV exports" />
    </div>
  );
}

// ── Step: Label Studio ────────────────────────────────────────────────────────

function StepLabelStudio({ state, set }: {
  state: WizardState;
  set: (f: keyof WizardState) => (v: string | boolean | number) => void;
}) {
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string; user?: string; projects?: number } | null>(null);
  const [creating, setCreating] = useState(false);
  const [createResult, setCreateResult] = useState<{ ok: boolean; msg: string; url?: string; id?: number } | null>(null);

  const testConnection = async () => {
    setTesting(true);
    setTestResult(null);
    setCreateResult(null);
    try {
      const res = await api.post('/setup/label-studio/test', {
        url: state.labelStudioUrl,
        api_key: state.labelStudioKey,
      });
      const d = res.data;
      if (!d.reachable) {
        setTestResult({ ok: false, msg: `Not reachable at ${state.labelStudioUrl}. Is Label Studio running?` });
      } else if (!d.authenticated) {
        setTestResult({ ok: false, msg: d.detail || 'Authentication failed. Check your API key.' });
      } else {
        setTestResult({ ok: true, msg: `Connected as ${d.user}`, user: d.user, projects: d.projects_count });
      }
    } catch (e: unknown) {
      setTestResult({ ok: false, msg: e instanceof Error ? e.message : 'Request failed' });
    } finally {
      setTesting(false);
    }
  };

  const createProject = async () => {
    setCreating(true);
    setCreateResult(null);
    try {
      const res = await api.post('/setup/label-studio/create-project', {
        url: state.labelStudioUrl,
        api_key: state.labelStudioKey,
        project_name: state.labelStudioProjectName,
      });
      const d = res.data;
      if (d.ok) {
        (set('labelStudioProjectId') as (v: number) => void)(d.project_id);
        setCreateResult({
          ok: true,
          msg: d.already_existed ? 'Project already exists — reusing it.' : 'Project created successfully!',
          url: d.project_url,
          id: d.project_id,
        });
      } else {
        setCreateResult({ ok: false, msg: d.detail || 'Failed to create project.' });
      }
    } catch (e: unknown) {
      setCreateResult({ ok: false, msg: e instanceof Error ? e.message : 'Request failed' });
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Label Studio Setup</h2>
        <p className="text-sm text-text-muted mt-1">
          Configure and test your annotation platform. CTIP will create a ready-to-use trichome detection project.
        </p>
      </div>

      <InfoBox>
        Label Studio runs on port 3005 (Docker). Start it with:{' '}
        <code>cd docker && docker compose --profile annotation up -d</code>
      </InfoBox>

      {/* Connection */}
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary border-b border-border pb-2">
          Connection
        </p>
        <InputField label="URL" value={state.labelStudioUrl}
          onChange={set('labelStudioUrl') as (v: string) => void}
          placeholder="http://localhost:3005" />
        <InputField label="API Key" value={state.labelStudioKey}
          onChange={set('labelStudioKey') as (v: string) => void}
          placeholder="••••••••••••••••••••••"
          hint="Label Studio → Account & Settings → Access Token"
          type="password" />
        <button onClick={testConnection} disabled={testing}
          className="btn-secondary w-full">
          {testing ? <><Loader2 className="w-4 h-4 animate-spin" />Testing…</> : 'Test Connection'}
        </button>
        {testResult && (
          <div className={['flex items-start gap-2 rounded-md p-3 text-xs',
            testResult.ok ? 'text-status-success' : 'text-status-error'].join(' ')}
            style={{ background: testResult.ok ? 'rgba(34,197,94,0.07)' : 'rgba(239,68,68,0.07)',
              border: `1px solid ${testResult.ok ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}` }}>
            {testResult.ok ? <CheckCircle className="w-4 h-4 shrink-0" /> : <XCircle className="w-4 h-4 shrink-0" />}
            <div>
              <p>{testResult.msg}</p>
              {testResult.ok && testResult.projects !== undefined && (
                <p className="text-text-muted mt-0.5">{testResult.projects} existing project(s)</p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Project setup */}
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary border-b border-border pb-2">
          Annotation Project
        </p>
        <InputField label="Project Name" value={state.labelStudioProjectName}
          onChange={set('labelStudioProjectName') as (v: string) => void}
          placeholder="CTIP — Trichome Detection" />

        <div className="rounded-md p-3 text-xs space-y-1.5" style={{ background: '#0d1117', border: '1px solid #21262d' }}>
          <p className="font-medium text-text-secondary">Label config included (4 classes + quality):</p>
          <div className="flex flex-wrap gap-2 mt-1">
            {[
              { name: 'stalked',       color: '#22d3ee' },
              { name: 'sessile',       color: '#34d399' },
              { name: 'bulbous',       color: '#a78bfa' },
              { name: 'non-glandular', color: '#fb923c' },
            ].map(l => (
              <span key={l.name} className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px]"
                style={{ background: l.color + '22', border: `1px solid ${l.color}55`, color: l.color }}>
                <span className="w-2 h-2 rounded-full" style={{ background: l.color }} />
                {l.name}
              </span>
            ))}
            <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] text-text-muted"
              style={{ border: '1px solid #21262d' }}>
              + quality rating + notes field
            </span>
          </div>
        </div>

        <button onClick={createProject}
          disabled={creating || !testResult?.ok}
          className="btn-primary w-full">
          {creating
            ? <><Loader2 className="w-4 h-4 animate-spin" />Creating project…</>
            : state.labelStudioProjectId
            ? <><CheckCircle2 className="w-4 h-4" />Project set up — re-create</>
            : 'Create Annotation Project'}
        </button>
        {!testResult?.ok && (
          <p className="text-xs text-text-muted text-center">Test connection first to enable project creation.</p>
        )}
        {createResult && (
          <div className={['flex items-start gap-2 rounded-md p-3 text-xs',
            createResult.ok ? 'text-status-success' : 'text-status-error'].join(' ')}
            style={{ background: createResult.ok ? 'rgba(34,197,94,0.07)' : 'rgba(239,68,68,0.07)',
              border: `1px solid ${createResult.ok ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}` }}>
            {createResult.ok ? <CheckCircle className="w-4 h-4 shrink-0" /> : <XCircle className="w-4 h-4 shrink-0" />}
            <div>
              <p>{createResult.msg}</p>
              {createResult.url && (
                <a href={createResult.url} target="_blank" rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 mt-1 text-accent hover:underline">
                  Open project <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Step: Services ────────────────────────────────────────────────────────────

function StepServices({ state, set, errors }: {
  state: WizardState;
  set: (f: keyof WizardState) => (v: string | boolean) => void;
  errors: Record<string, string>;
}) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Experiment Tracking</h2>
        <p className="text-sm text-text-muted mt-1">Connect MLflow and optionally Weights &amp; Biases.</p>
      </div>
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary border-b border-border pb-2">MLflow</p>
        <InputField label="Tracking URI" value={state.mlflowUri}
          onChange={set('mlflowUri') as (v: string) => void}
          placeholder="http://localhost:3004" error={errors.mlflowUri} />
        <InputField label="Default Experiment" value={state.mlflowExperiment}
          onChange={set('mlflowExperiment') as (v: string) => void}
          placeholder="trichome-detection" />
      </div>
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary border-b border-border pb-2">Weights &amp; Biases (optional)</p>
        <Toggle label="Enable W&B logging" hint="Requires a wandb.ai account" value={state.useWandb} onChange={set('useWandb')} />
        {state.useWandb && (
          <>
            <InputField label="API Key" value={state.wandbKey}
              onChange={set('wandbKey') as (v: string) => void} type="password" />
            <InputField label="Project" value={state.wandbProject}
              onChange={set('wandbProject') as (v: string) => void} placeholder="trichome-detection" />
          </>
        )}
      </div>
    </div>
  );
}

// ── Step: Security ────────────────────────────────────────────────────────────

function StepSecurity({ state, set }: {
  state: WizardState;
  set: (f: keyof WizardState) => (v: string | boolean) => void;
}) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Security</h2>
        <p className="text-sm text-text-muted mt-1">Secret key for session tokens and optional API authentication.</p>
      </div>
      <WarnBox>If CTIP is internet-facing, set a strong API_TOKEN. Empty = no auth (dev mode only).</WarnBox>
      <InputField label="Secret Key" value={state.secretKey}
        onChange={set('secretKey') as (v: string) => void}
        placeholder="Leave empty to keep existing value"
        type="password"
        action={{ label: 'Generate random', onClick: () => (set('secretKey') as (v: string) => void)(generateSecretKey()) }} />
      <InputField label="API Token (optional)" value={state.apiToken}
        onChange={set('apiToken') as (v: string) => void}
        placeholder="Leave empty to disable auth"
        hint="Authorization: Bearer <token> required on all API requests"
        type="password" />
      <InfoBox>Single-user bearer token model. For multi-user access, configure nginx auth at the network layer.</InfoBox>
    </div>
  );
}

// ── Step: Review ──────────────────────────────────────────────────────────────

function StepReview({ state }: { state: WizardState }) {
  const sections = [
    { title: 'Network', rows: [
      { k: 'PUBLIC_DOMAIN', v: state.publicDomain || '(localhost only)' },
      { k: 'PUBLIC_PORT', v: state.publicPort },
    ]},
    { title: 'Hardware', rows: [
      { k: 'CUDA_DEVICE', v: state.cudaDevice },
      { k: 'VRAM_LIMIT_GB', v: state.vramLimit },
      { k: 'VRAM_INFERENCE_BUDGET_GB', v: state.vramInference },
      { k: 'ENVIRONMENT', v: state.environment },
    ]},
    { title: 'Storage', rows: [
      { k: 'DATA_ROOT', v: state.dataRoot },
      { k: 'MODELS_DIR', v: state.modelsDir },
      { k: 'OUTPUTS_DIR', v: state.outputsDir },
    ]},
    { title: 'Label Studio', rows: [
      { k: 'LABEL_STUDIO_URL', v: state.labelStudioUrl },
      { k: 'LABEL_STUDIO_API_KEY', v: state.labelStudioKey, s: true },
      { k: 'LS Project', v: state.labelStudioProjectId ? `ID ${state.labelStudioProjectId} — ${state.labelStudioProjectName}` : '(not created yet)' },
    ]},
    { title: 'Services', rows: [
      { k: 'MLFLOW_TRACKING_URI', v: state.mlflowUri },
      { k: 'MLFLOW_EXPERIMENT_NAME', v: state.mlflowExperiment },
      { k: 'USE_WANDB', v: state.useWandb ? 'true' : 'false' },
    ]},
    { title: 'Security', rows: [
      { k: 'SECRET_KEY', v: state.secretKey || '(unchanged)', s: true },
      { k: 'API_TOKEN', v: state.apiToken || '(disabled)', s: true },
    ]},
  ];

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Review Configuration</h2>
        <p className="text-sm text-text-muted mt-1">
          Click <strong>Save &amp; Finish</strong> to write to <code className="text-accent text-xs">.env</code>.
          A live verification runs automatically after saving.
        </p>
      </div>
      {sections.map(sec => (
        <div key={sec.title}>
          <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1.5">{sec.title}</p>
          <div className="rounded-lg overflow-hidden" style={{ border: '1px solid #21262d' }}>
            {sec.rows.map(row => (
              <div key={row.k} className="flex items-start gap-3 px-3 py-2 border-b border-border last:border-0">
                <span className="text-xs text-text-muted min-w-[180px] shrink-0 pt-0.5">{row.k}</span>
                <span className={['text-xs font-mono break-all', row.v ? 'text-text-primary' : 'text-text-muted italic'].join(' ')}>
                  {row.v ? (row.s ? '••••••••' : row.v) : '(not set)'}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
      <InfoBox>Existing .env keys not covered by this wizard are left untouched.</InfoBox>
    </div>
  );
}

// ── Step: Verification (post-save) ────────────────────────────────────────────

function StepVerification({ onDone }: { onDone: () => void }) {
  const [items, setItems] = useState<VerificationItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [ran, setRan] = useState(false);
  const [allOk, setAllOk] = useState<boolean | null>(null);

  const run = async () => {
    setLoading(true);
    setItems([]);
    try {
      const res = await api.get('/setup/verification');
      setItems(res.data.items);
      setAllOk(res.data.all_ok);
    } catch {
      setItems([{ name: 'API', url: '', ok: false, status_code: 0, latency_ms: 0, detail: 'Backend unreachable' }]);
      setAllOk(false);
    } finally {
      setLoading(false);
      setRan(true);
    }
  };

  useEffect(() => { run(); }, []);

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-text-primary">Verification</h2>
          <p className="text-sm text-text-muted mt-1">Live health check of all configured subsystems.</p>
        </div>
        <button onClick={run} disabled={loading} className="btn-secondary text-xs flex items-center gap-1.5">
          <RefreshCw className={['w-3.5 h-3.5', loading ? 'animate-spin' : ''].join(' ')} />
          Re-run
        </button>
      </div>

      {loading && items.length === 0 && (
        <div className="flex items-center gap-3 py-8 justify-center text-text-muted">
          <Loader2 className="w-5 h-5 animate-spin text-accent" />
          <span className="text-sm">Checking all services…</span>
        </div>
      )}

      {items.length > 0 && (
        <div className="rounded-lg overflow-hidden font-mono text-xs"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}>
          <div className="flex items-center gap-2 px-3 py-2 border-b border-border"
            style={{ background: '#161b22' }}>
            <Terminal className="w-3.5 h-3.5 text-text-muted" />
            <span className="text-text-muted">ctip verification log</span>
            <span className="ml-auto text-text-muted">{new Date().toLocaleTimeString()}</span>
          </div>
          <div className="p-3 space-y-1.5">
            {items.map((item, i) => (
              <div key={i} className="flex items-center gap-3">
                <StatusDot ok={item.ok} />
                <span className={item.ok ? 'text-status-success' : 'text-status-error'}>
                  {item.ok ? 'PASS' : 'FAIL'}
                </span>
                <span className="text-text-secondary">{item.name}</span>
                <span className="text-text-muted ml-auto flex items-center gap-2">
                  {item.ok && item.status_code > 0 && (
                    <span className="text-status-success">HTTP {item.status_code}</span>
                  )}
                  {item.ok && (
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />{item.latency_ms}ms
                    </span>
                  )}
                  {!item.ok && item.detail && (
                    <span className="text-status-error truncate max-w-[200px]">{item.detail}</span>
                  )}
                </span>
              </div>
            ))}
            <div className="border-t border-border mt-2 pt-2">
              <span className={allOk ? 'text-status-success' : 'text-status-warning'}>
                {allOk ? '✓ All systems operational' : '⚠ Some services unavailable — check logs'}
              </span>
            </div>
          </div>
        </div>
      )}

      {ran && (
        <div className="flex gap-3">
          <button className="btn-primary flex-1" onClick={onDone}>
            <CheckCircle2 className="w-4 h-4" />
            Go to Dashboard
          </button>
          <button className="btn-secondary" onClick={run} disabled={loading}>
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function SetupPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [state, dispatch] = useReducer(reducer, DEFAULT_STATE);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState('');
  const [saved, setSaved] = useState(false);
  const [loadingConfig, setLoadingConfig] = useState(true);

  const stepId = STEPS[currentStep]?.id as StepId | undefined;

  // Load existing config on mount
  useEffect(() => {
    (async () => {
      try {
        const draft = sessionStorage.getItem(SESSION_KEY);
        if (draft) {
          dispatch({ type: 'LOAD', payload: JSON.parse(draft) });
        } else {
          const res = await api.get('/setup/config');
          const map: Record<string, string> = {};
          for (const e of res.data.entries as { key: string; value: string }[]) map[e.key] = e.value;
          dispatch({ type: 'LOAD', payload: {
            publicDomain: map['PUBLIC_DOMAIN'] ?? '',
            publicPort: map['PUBLIC_PORT'] ?? '3001',
            cudaDevice: map['CUDA_DEVICE'] ?? 'cuda:0',
            cudaVisible: map['CUDA_VISIBLE_DEVICES'] ?? '0',
            vramLimit: map['VRAM_LIMIT_GB'] ?? '8.0',
            vramInference: map['VRAM_INFERENCE_BUDGET_GB'] ?? '2.0',
            dataRoot: map['DATA_ROOT'] ?? DEFAULT_STATE.dataRoot,
            modelsDir: map['MODELS_DIR'] ?? DEFAULT_STATE.modelsDir,
            outputsDir: map['OUTPUTS_DIR'] ?? DEFAULT_STATE.outputsDir,
            labelStudioUrl: map['LABEL_STUDIO_URL'] ?? 'http://localhost:3005',
            mlflowUri: map['MLFLOW_TRACKING_URI'] ?? 'http://localhost:3004',
            mlflowExperiment: map['MLFLOW_EXPERIMENT_NAME'] ?? 'trichome-detection',
            useWandb: map['USE_WANDB'] === 'true',
            wandbProject: map['WANDB_PROJECT'] ?? 'trichome-detection',
            environment: map['ENVIRONMENT'] ?? 'development',
          }});
        }
      } catch { /* offline */ } finally { setLoadingConfig(false); }
    })();
  }, []);

  // Persist draft
  useEffect(() => {
    if (!loadingConfig) sessionStorage.setItem(SESSION_KEY, JSON.stringify(state));
  }, [state, loadingConfig]);

  const set = useCallback(
    (field: keyof WizardState) => (value: string | boolean | number) => {
      dispatch({ type: 'SET', field, value });
      setErrors(e => { const n = { ...e }; delete n[field]; return n; });
    }, []);

  function goNext() {
    if (!stepId) return;
    const errs = validateStep(stepId, state);
    if (Object.keys(errs).length) { setErrors(errs); return; }
    setErrors({});
    setCurrentStep(s => Math.min(s + 1, STEPS.length - 1));
  }

  function goBack() {
    setCurrentStep(s => Math.max(s - 1, 0));
    setErrors({});
  }

  async function handleSubmit() {
    setSubmitting(true);
    setSubmitError('');
    try {
      await api.post('/setup/configure', {
        settings: stateToEnvMap(state),
        mark_setup_complete: true,
      });
      sessionStorage.removeItem(SESSION_KEY);
      sessionStorage.setItem('ctip-setup-checked', '1');
      setSaved(true);
      // Advance to verification step
      setCurrentStep(STEPS.length); // past last step = verification screen
    } catch (e: unknown) {
      setSubmitError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setSubmitting(false);
    }
  }

  if (loadingConfig) {
    return (
      <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-background">
        <Loader2 className="w-8 h-8 text-accent animate-spin" />
      </div>
    );
  }

  const isReviewStep = currentStep === STEPS.length - 1;
  const isVerificationScreen = saved || currentStep >= STEPS.length;

  return (
    <div className="fixed inset-0 z-[9999] bg-background flex overflow-hidden">

      {/* ── Left panel: step list ─────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col w-64 bg-surface border-r border-border py-10 px-6 shrink-0">
        <div className="flex items-center gap-3 mb-10">
          <Microscope className="w-7 h-7 text-accent" />
          <div>
            <p className="text-sm font-semibold text-text-primary">CTIP</p>
            <p className="text-xs text-text-muted">Full Installer</p>
          </div>
        </div>
        <nav className="space-y-1">
          {STEPS.map((step, idx) => {
            const Icon = step.icon;
            const isActive = idx === currentStep && !isVerificationScreen;
            const isDone = idx < currentStep || isVerificationScreen;
            return (
              <div key={step.id} className={['flex items-center gap-3 px-3 py-2.5 rounded-md text-sm',
                isActive ? 'bg-accent/15 text-accent font-medium'
                  : isDone ? 'text-text-secondary' : 'text-text-muted'].join(' ')}>
                <div className={['w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 border',
                  isActive ? 'border-accent bg-accent text-white'
                    : isDone ? 'border-accent bg-accent/20 text-accent'
                    : 'border-border bg-panel text-text-muted'].join(' ')}>
                  {isDone ? '✓' : idx + 1}
                </div>
                <span>{step.label}</span>
              </div>
            );
          })}
          {/* Verification pseudo-step */}
          <div className={['flex items-center gap-3 px-3 py-2.5 rounded-md text-sm',
            isVerificationScreen ? 'bg-accent/15 text-accent font-medium' : 'text-text-muted'].join(' ')}>
            <div className={['w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 border',
              isVerificationScreen ? 'border-accent bg-accent text-white' : 'border-border bg-panel text-text-muted'].join(' ')}>
              {isVerificationScreen ? '✓' : STEPS.length + 1}
            </div>
            <span>Verification</span>
          </div>
        </nav>
        <div className="mt-auto text-xs text-text-muted leading-relaxed">
          Re-run anytime from sidebar: <span className="text-text-secondary">First-Time Setup</span>
        </div>
      </aside>

      {/* ── Right panel: step content ─────────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-y-auto">
        <div className="lg:hidden h-1 bg-panel">
          <div className="h-1 bg-accent transition-all duration-500"
            style={{ width: `${((currentStep + 1) / (STEPS.length + 1)) * 100}%` }} />
        </div>

        <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
          <div className="w-full max-w-xl">

            <div key={currentStep} className="space-y-8 animate-wizard-step">
              {currentStep === 0 && <StepWelcome />}
              {currentStep === 1 && <StepSystemCheck />}
              {currentStep === 2 && <StepNetwork state={state} set={set} errors={errors} />}
              {currentStep === 3 && <StepHardware state={state} set={set} errors={errors} />}
              {currentStep === 4 && <StepStorage state={state} set={set} errors={errors} />}
              {currentStep === 5 && <StepLabelStudio state={state} set={set} />}
              {currentStep === 6 && <StepServices state={state} set={set} errors={errors} />}
              {currentStep === 7 && <StepSecurity state={state} set={set} />}
              {currentStep === 8 && <StepReview state={state} />}
              {isVerificationScreen && (
                <StepVerification onDone={() => router.replace('/')} />
              )}
            </div>

            {/* Navigation */}
            {!isVerificationScreen && (
              <div className="flex items-center justify-between mt-10 pt-6 border-t border-border">
                <button className="btn-secondary" onClick={goBack} disabled={currentStep === 0}>
                  <ChevronLeft className="w-4 h-4" />Back
                </button>
                <span className="text-xs text-text-muted lg:hidden">{currentStep + 1} / {STEPS.length}</span>
                {!isReviewStep ? (
                  <button className="btn-primary" onClick={goNext}>
                    {currentStep === 0 ? 'Start Check' : 'Continue'}
                    <ChevronRight className="w-4 h-4" />
                  </button>
                ) : (
                  <button className="btn-primary" onClick={handleSubmit} disabled={submitting}>
                    {submitting
                      ? <><Loader2 className="w-4 h-4 animate-spin" />Saving…</>
                      : <><CheckCircle2 className="w-4 h-4" />Save & Finish</>}
                  </button>
                )}
              </div>
            )}
            {submitError && <p className="mt-4 text-xs text-red-400 text-center">{submitError}</p>}
          </div>
        </div>
      </main>
    </div>
  );
}
