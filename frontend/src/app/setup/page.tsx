'use client';

/**
 * CTIP First-Time Setup Wizard
 *
 * OS-style multi-step assistant (macOS Setup / Windows OOBE inspired).
 * Covers: Network → Hardware → Storage → External Services → Security → Review → Done.
 *
 * Architecture:
 * - Pure client-side state machine (no route segments per step).
 * - Persists draft state in sessionStorage so a page refresh doesn't lose work.
 * - Calls POST /api/v1/setup/configure on final confirm.
 * - Reads current config from GET /api/v1/setup/config on mount.
 */

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Microscope,
  Globe,
  Cpu,
  HardDrive,
  Plug,
  ShieldCheck,
  CheckCircle2,
  ChevronRight,
  ChevronLeft,
  Loader2,
  AlertTriangle,
  Info,
  Eye,
  EyeOff,
  ExternalLink,
  RotateCcw,
} from 'lucide-react';
import { api } from '@/lib/api';

// ── Types ─────────────────────────────────────────────────────────────────────

interface WizardState {
  // Network
  publicDomain: string;
  publicPort: string;
  // Hardware
  cudaDevice: string;
  cudaVisible: string;
  vramLimit: string;
  vramInference: string;
  // Storage
  dataRoot: string;
  modelsDir: string;
  outputsDir: string;
  // Services
  labelStudioUrl: string;
  labelStudioKey: string;
  mlflowUri: string;
  mlflowExperiment: string;
  useWandb: boolean;
  wandbKey: string;
  wandbProject: string;
  // Security
  secretKey: string;
  apiToken: string;
  // App
  environment: string;
}

interface ValidationErrors {
  [field: string]: string;
}

type WizardAction =
  | { type: 'SET'; field: keyof WizardState; value: string | boolean }
  | { type: 'LOAD'; payload: Partial<WizardState> };

// ── Constants ─────────────────────────────────────────────────────────────────

const STEPS = [
  { id: 'welcome', label: 'Welcome', icon: Microscope },
  { id: 'network', label: 'Network', icon: Globe },
  { id: 'hardware', label: 'Hardware', icon: Cpu },
  { id: 'storage', label: 'Storage', icon: HardDrive },
  { id: 'services', label: 'Services', icon: Plug },
  { id: 'security', label: 'Security', icon: ShieldCheck },
  { id: 'review', label: 'Review', icon: CheckCircle2 },
] as const;

type StepId = (typeof STEPS)[number]['id'];

const DEFAULT_STATE: WizardState = {
  publicDomain: '',
  publicPort: '3001',
  cudaDevice: 'cuda:0',
  cudaVisible: '0',
  vramLimit: '8.0',
  vramInference: '2.0',
  dataRoot: '/path/to/trichome-analysis/data',
  modelsDir: '/path/to/trichome-analysis/data/models',
  outputsDir: '/path/to/trichome-analysis/data/outputs',
  labelStudioUrl: 'http://localhost:3005',
  labelStudioKey: '',
  mlflowUri: 'http://localhost:3004',
  mlflowExperiment: 'trichome-detection',
  useWandb: false,
  wandbKey: '',
  wandbProject: 'trichome-detection',
  secretKey: '',
  apiToken: '',
  environment: 'development',
};

const SESSION_KEY = 'ctip-setup-draft';

// ── Reducer ───────────────────────────────────────────────────────────────────

function wizardReducer(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case 'SET':
      return { ...state, [action.field]: action.value };
    case 'LOAD':
      return { ...state, ...action.payload };
    default:
      return state;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function generateSecretKey(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
  return Array.from({ length: 64 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

function validateStep(step: StepId, state: WizardState): ValidationErrors {
  const errors: ValidationErrors = {};

  if (step === 'network') {
    if (state.publicDomain) {
      const domainRe = /^[a-zA-Z0-9]([a-zA-Z0-9\-.]{0,253}[a-zA-Z0-9])?$/;
      if (!domainRe.test(state.publicDomain)) {
        errors.publicDomain = 'Invalid domain format (e.g. mylab.ddns.net)';
      }
    }
    const port = parseInt(state.publicPort, 10);
    if (isNaN(port) || port < 1 || port > 65535) {
      errors.publicPort = 'Port must be 1–65535';
    }
  }

  if (step === 'hardware') {
    const vram = parseFloat(state.vramLimit);
    if (isNaN(vram) || vram < 1 || vram > 80) {
      errors.vramLimit = 'VRAM must be between 1 and 80 GB';
    }
    const inf = parseFloat(state.vramInference);
    if (isNaN(inf) || inf < 0) {
      errors.vramInference = 'Must be ≥ 0';
    }
  }

  if (step === 'storage') {
    if (!state.dataRoot.startsWith('/')) {
      errors.dataRoot = 'Must be an absolute path starting with /';
    }
  }

  if (step === 'services') {
    const urlRe = /^https?:\/\//;
    if (state.labelStudioUrl && !urlRe.test(state.labelStudioUrl)) {
      errors.labelStudioUrl = 'Must be a valid HTTP URL';
    }
    if (state.mlflowUri && !urlRe.test(state.mlflowUri)) {
      errors.mlflowUri = 'Must be a valid HTTP URL';
    }
  }

  return errors;
}

function stateToEnvMap(s: WizardState): Record<string, string> {
  return {
    PUBLIC_DOMAIN: s.publicDomain,
    PUBLIC_PORT: s.publicPort,
    CUDA_DEVICE: s.cudaDevice,
    CUDA_VISIBLE_DEVICES: s.cudaVisible,
    VRAM_LIMIT_GB: s.vramLimit,
    VRAM_INFERENCE_BUDGET_GB: s.vramInference,
    DATA_ROOT: s.dataRoot,
    MODELS_DIR: s.modelsDir,
    OUTPUTS_DIR: s.outputsDir,
    LABEL_STUDIO_URL: s.labelStudioUrl,
    LABEL_STUDIO_API_KEY: s.labelStudioKey,
    MLFLOW_TRACKING_URI: s.mlflowUri,
    MLFLOW_EXPERIMENT_NAME: s.mlflowExperiment,
    USE_WANDB: s.useWandb ? 'true' : 'false',
    WANDB_API_KEY: s.wandbKey,
    WANDB_PROJECT: s.wandbProject,
    SECRET_KEY: s.secretKey,
    API_TOKEN: s.apiToken,
    ENVIRONMENT: s.environment,
  };
}

// ── Sub-components ────────────────────────────────────────────────────────────

function InputField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  error,
  type = 'text',
  monospace = false,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  hint?: string;
  error?: string;
  type?: string;
  monospace?: boolean;
}) {
  const [showPass, setShowPass] = useState(false);
  const isPassword = type === 'password';
  const inputType = isPassword ? (showPass ? 'text' : 'password') : type;

  return (
    <div className="space-y-1">
      <label className="block text-xs font-medium text-text-secondary uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <input
          type={inputType}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={[
            'input pr-10',
            monospace ? 'font-mono text-xs' : '',
            error ? 'border-red-500 focus:border-red-500 focus:shadow-[0_0_0_1px_#ef4444]' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          spellCheck={false}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPass((p) => !p)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary transition-colors"
          >
            {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        )}
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {hint && !error && <p className="text-xs text-text-muted">{hint}</p>}
    </div>
  );
}

function Toggle({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <div>
        <p className="text-sm text-text-primary">{label}</p>
        {hint && <p className="text-xs text-text-muted mt-0.5">{hint}</p>}
      </div>
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={[
          'relative inline-flex h-6 w-11 shrink-0 rounded-full border-2 border-transparent transition-colors duration-200',
          value ? 'bg-accent' : 'bg-border',
        ].join(' ')}
        role="switch"
        aria-checked={value}
      >
        <span
          className={[
            'pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform transition-transform duration-200',
            value ? 'translate-x-5' : 'translate-x-0',
          ].join(' ')}
        />
      </button>
    </div>
  );
}

function InfoBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-md p-3 text-xs"
      style={{
        background: 'rgba(59,130,246,0.07)',
        border: '1px solid rgba(59,130,246,0.2)',
        color: '#60a5fa',
      }}>
      <Info className="w-4 h-4 mt-0.5 shrink-0" />
      <span>{children}</span>
    </div>
  );
}

function WarnBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-md p-3 text-xs"
      style={{
        background: 'rgba(234,179,8,0.07)',
        border: '1px solid rgba(234,179,8,0.2)',
        color: '#eab308',
      }}>
      <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
      <span>{children}</span>
    </div>
  );
}

function ReviewRow({ label, value, sensitive = false }: { label: string; value: string; sensitive?: boolean }) {
  return (
    <div className="flex items-start gap-3 py-2 border-b border-border last:border-0">
      <span className="text-xs text-text-muted min-w-[180px] shrink-0 pt-0.5">{label}</span>
      <span
        className={[
          'text-xs font-mono break-all',
          value ? 'text-text-primary' : 'text-text-muted italic',
        ].join(' ')}
      >
        {!value
          ? '(not set)'
          : sensitive
          ? '••••••••'
          : value}
      </span>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function SetupPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [direction, setDirection] = useState<'forward' | 'back'>('forward');
  const [state, dispatch] = useReducer(wizardReducer, DEFAULT_STATE);
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState('');
  const [done, setDone] = useState(false);
  const [loadingConfig, setLoadingConfig] = useState(true);
  const prevStep = useRef(0);

  const stepId = STEPS[currentStep]?.id as StepId | undefined;

  // ── Load existing config on mount ──────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        // Try to restore draft from session
        const draft = sessionStorage.getItem(SESSION_KEY);
        if (draft) {
          dispatch({ type: 'LOAD', payload: JSON.parse(draft) });
        } else {
          // Load from backend
          const res = await api.get('/setup/config');
          const map: Record<string, string> = {};
          for (const entry of res.data.entries as { key: string; value: string }[]) {
            map[entry.key] = entry.value;
          }
          const partial: Partial<WizardState> = {
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
          };
          dispatch({ type: 'LOAD', payload: partial });
        }
      } catch {
        // Offline / backend not started yet — use defaults
      } finally {
        setLoadingConfig(false);
      }
    })();
  }, []);

  // ── Persist draft to sessionStorage ────────────────────────────────
  useEffect(() => {
    if (!loadingConfig) {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(state));
    }
  }, [state, loadingConfig]);

  // ── Navigation ─────────────────────────────────────────────────────
  const set = useCallback(
    (field: keyof WizardState) => (value: string | boolean) => {
      dispatch({ type: 'SET', field, value });
      setErrors((e) => {
        const next = { ...e };
        delete next[field];
        return next;
      });
    },
    [],
  );

  function goNext() {
    if (!stepId) return;
    const errs = validateStep(stepId, state);
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }
    setErrors({});
    prevStep.current = currentStep;
    setDirection('forward');
    setCurrentStep((s) => Math.min(s + 1, STEPS.length - 1));
  }

  function goBack() {
    prevStep.current = currentStep;
    setDirection('back');
    setCurrentStep((s) => Math.max(s - 1, 0));
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
      setDone(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setSubmitError(msg);
    } finally {
      setSubmitting(false);
    }
  }

  // ── Completion screen ───────────────────────────────────────────────
  if (done) {
    return (
      <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-background overflow-y-auto">
        <div className="text-center space-y-6 max-w-md px-6">
          <div className="flex justify-center">
            <div className="w-20 h-20 rounded-full flex items-center justify-center"
              style={{ background: 'rgba(35,134,54,0.15)', border: '2px solid #238636' }}>
              <CheckCircle2 className="w-10 h-10 text-accent" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-text-primary">Setup Complete</h1>
          <p className="text-text-secondary text-sm leading-relaxed">
            Your CTIP configuration has been saved to <code className="text-accent">.env</code>.
            Restart Docker Compose to apply changes.
          </p>
          <div className="code-block text-left">
            <span className="text-text-muted">$ </span>
            <span className="text-accent">cd docker && docker compose down && docker compose up -d</span>
          </div>
          <div className="flex gap-3 justify-center">
            <button
              className="btn-primary"
              onClick={() => router.push('/')}
            >
              Go to Dashboard
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              className="btn-secondary"
              onClick={() => { setDone(false); setCurrentStep(0); }}
            >
              <RotateCcw className="w-4 h-4" />
              Run Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (loadingConfig) {
    return (
      <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-background">
        <Loader2 className="w-8 h-8 text-accent animate-spin" />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-[9999] bg-background flex overflow-hidden">
      {/* ── Left panel: step list ─────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col w-64 bg-surface border-r border-border py-10 px-6">
        {/* Logo */}
        <div className="flex items-center gap-3 mb-10">
          <Microscope className="w-7 h-7 text-accent" />
          <div>
            <p className="text-sm font-semibold text-text-primary">CTIP</p>
            <p className="text-xs text-text-muted">First-Time Setup</p>
          </div>
        </div>

        {/* Steps */}
        <nav className="space-y-1">
          {STEPS.map((step, idx) => {
            const Icon = step.icon;
            const isActive = idx === currentStep;
            const isDone = idx < currentStep;

            return (
              <div
                key={step.id}
                className={[
                  'flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors',
                  isActive
                    ? 'bg-accent/15 text-accent font-medium'
                    : isDone
                    ? 'text-text-secondary'
                    : 'text-text-muted',
                ].join(' ')}
              >
                <div
                  className={[
                    'w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 border',
                    isActive
                      ? 'border-accent bg-accent text-white'
                      : isDone
                      ? 'border-accent bg-accent/20 text-accent'
                      : 'border-border bg-panel text-text-muted',
                  ].join(' ')}
                >
                  {isDone ? '✓' : idx + 1}
                </div>
                <span>{step.label}</span>
              </div>
            );
          })}
        </nav>

        {/* Footer note */}
        <div className="mt-auto text-xs text-text-muted leading-relaxed">
          You can re-run this wizard anytime from{' '}
          <span className="text-text-secondary">System → Setup</span>.
        </div>
      </aside>

      {/* ── Right panel: step content ─────────────────────────────── */}
      <main className="flex-1 flex flex-col min-h-screen overflow-y-auto">
        {/* Progress bar (mobile) */}
        <div className="lg:hidden h-1 bg-panel">
          <div
            className="h-1 bg-accent transition-all duration-500"
            style={{ width: `${((currentStep + 1) / STEPS.length) * 100}%` }}
          />
        </div>

        <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
          <div className="w-full max-w-xl">
            {/* Step content */}
            <div key={currentStep} className="space-y-8 animate-wizard-step">
              {currentStep === 0 && <StepWelcome />}
              {currentStep === 1 && <StepNetwork state={state} set={set} errors={errors} />}
              {currentStep === 2 && <StepHardware state={state} set={set} errors={errors} />}
              {currentStep === 3 && <StepStorage state={state} set={set} errors={errors} />}
              {currentStep === 4 && <StepServices state={state} set={set} errors={errors} />}
              {currentStep === 5 && <StepSecurity state={state} set={set} errors={errors} />}
              {currentStep === 6 && <StepReview state={state} />}
            </div>

            {/* Navigation */}
            <div className="flex items-center justify-between mt-10 pt-6 border-t border-border">
              <button
                className="btn-secondary"
                onClick={goBack}
                disabled={currentStep === 0}
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </button>

              {/* Mobile step indicator */}
              <span className="text-xs text-text-muted lg:hidden">
                {currentStep + 1} / {STEPS.length}
              </span>

              {currentStep < STEPS.length - 1 ? (
                <button className="btn-primary" onClick={goNext}>
                  {currentStep === 0 ? 'Get Started' : 'Continue'}
                  <ChevronRight className="w-4 h-4" />
                </button>
              ) : (
                <button
                  className="btn-primary"
                  onClick={handleSubmit}
                  disabled={submitting}
                >
                  {submitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving…
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-4 h-4" />
                      Save & Finish
                    </>
                  )}
                </button>
              )}
            </div>

            {submitError && (
              <p className="mt-4 text-xs text-red-400 text-center">{submitError}</p>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

// ── Step Components ───────────────────────────────────────────────────────────

function StepWelcome() {
  return (
    <div className="text-center space-y-6">
      <div className="flex justify-center">
        <div
          className="w-24 h-24 rounded-2xl flex items-center justify-center"
          style={{
            background: 'linear-gradient(135deg, rgba(35,134,54,0.2) 0%, rgba(59,130,246,0.15) 100%)',
            border: '1px solid rgba(35,134,54,0.4)',
          }}
        >
          <Microscope className="w-12 h-12 text-accent" />
        </div>
      </div>

      <div>
        <h1 className="text-3xl font-bold text-text-primary mb-2">
          Welcome to CTIP
        </h1>
        <p className="text-text-muted text-sm">
          Cannabis Trichome Intelligence Platform
        </p>
      </div>

      <p className="text-text-secondary text-sm leading-relaxed">
        This wizard will configure your platform in a few steps — network access, GPU settings,
        storage paths, and external service connections. All settings are saved to your{' '}
        <code className="text-accent text-xs">.env</code> file and can be changed anytime.
      </p>

      <div className="grid grid-cols-2 gap-3 text-left">
        {[
          { icon: Globe, label: 'Network', desc: 'Domain & public access' },
          { icon: Cpu, label: 'Hardware', desc: 'GPU & VRAM budget' },
          { icon: HardDrive, label: 'Storage', desc: 'Data directories' },
          { icon: Plug, label: 'Services', desc: 'Label Studio · MLflow' },
        ].map(({ icon: Icon, label, desc }) => (
          <div
            key={label}
            className="flex items-center gap-3 rounded-lg p-3"
            style={{ background: '#161b22', border: '1px solid #21262d' }}
          >
            <Icon className="w-4 h-4 text-accent shrink-0" />
            <div>
              <p className="text-xs font-medium text-text-primary">{label}</p>
              <p className="text-xs text-text-muted">{desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StepNetwork({
  state,
  set,
  errors,
}: {
  state: WizardState;
  set: (field: keyof WizardState) => (v: string | boolean) => void;
  errors: ValidationErrors;
}) {
  const isPublic = state.publicDomain.trim().length > 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Network Access</h2>
        <p className="text-sm text-text-muted mt-1">
          Configure how CTIP is accessible over the network.
        </p>
      </div>

      <InfoBox>
        By default CTIP runs on <strong>localhost only</strong> — not reachable from outside your
        machine. Enter a domain below to enable public access via nginx.
      </InfoBox>

      <div className="space-y-4">
        <div>
          <Toggle
            label="Enable public access"
            hint="Expose the platform at a DDNS/static domain via nginx reverse proxy."
            value={isPublic}
            onChange={(v) => {
              if (!v) set('publicDomain')('');
            }}
          />
        </div>

        {isPublic && (
          <InputField
            label="Public Domain"
            value={state.publicDomain}
            onChange={set('publicDomain') as (v: string) => void}
            placeholder="mylab.ddns.net"
            hint="No protocol prefix — just the hostname (or IP)."
            error={errors.publicDomain}
          />
        )}

        <InputField
          label="Public Port"
          value={state.publicPort}
          onChange={set('publicPort') as (v: string) => void}
          placeholder="3001"
          hint="nginx listens on this host port. Default: 3001."
          error={errors.publicPort}
        />
      </div>

      {!isPublic && (
        <div className="text-xs text-text-muted space-y-1 rounded-md p-3"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}>
          <p className="font-medium text-text-secondary">Access URLs (localhost only)</p>
          <p>
            <span className="text-text-muted">Frontend:</span>{' '}
            <a href={`http://localhost:${state.publicPort}`} target="_blank" rel="noopener noreferrer"
              className="text-accent hover:underline inline-flex items-center gap-1">
              http://localhost:{state.publicPort} <ExternalLink className="w-3 h-3" />
            </a>
          </p>
          <p>
            <span className="text-text-muted">API:</span>{' '}
            <span className="text-accent">http://localhost:3002/api/v1</span>
          </p>
        </div>
      )}

      {isPublic && state.publicDomain && !errors.publicDomain && (
        <div className="text-xs text-text-muted space-y-1 rounded-md p-3"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}>
          <p className="font-medium text-text-secondary">Access URLs (public)</p>
          <p>
            <span className="text-text-muted">Frontend:</span>{' '}
            <span className="text-accent">http://{state.publicDomain}:{state.publicPort}</span>
          </p>
        </div>
      )}

      <WarnBox>
        After changing network settings, restart Docker Compose:{' '}
        <code>cd docker && docker compose down && docker compose up -d</code>
      </WarnBox>
    </div>
  );
}

function StepHardware({
  state,
  set,
  errors,
}: {
  state: WizardState;
  set: (field: keyof WizardState) => (v: string | boolean) => void;
  errors: ValidationErrors;
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">GPU & Hardware</h2>
        <p className="text-sm text-text-muted mt-1">
          Configure CUDA device and VRAM budgets. Optimized for RTX 4060 (8 GB).
        </p>
      </div>

      <InfoBox>
        CTIP uses exactly <strong>one GPU task at a time</strong> (asyncio.Semaphore(1)).
        Set VRAM_LIMIT_GB to your card&apos;s total VRAM; VRAM_INFERENCE_BUDGET_GB is reserved
        for inference while training is running.
      </InfoBox>

      <div className="space-y-4">
        <InputField
          label="CUDA Device"
          value={state.cudaDevice}
          onChange={set('cudaDevice') as (v: string) => void}
          placeholder="cuda:0"
          hint="Device string passed to PyTorch. Use cuda:0 for the primary GPU, cpu for CPU-only."
        />

        <InputField
          label="CUDA_VISIBLE_DEVICES"
          value={state.cudaVisible}
          onChange={set('cudaVisible') as (v: string) => void}
          placeholder="0"
          hint="Which GPU indices are visible to the process (comma-separated). Usually just '0'."
        />

        <div className="grid grid-cols-2 gap-4">
          <InputField
            label="Total VRAM (GB)"
            value={state.vramLimit}
            onChange={set('vramLimit') as (v: string) => void}
            placeholder="8.0"
            hint="Total card VRAM. RTX 4060 = 8.0"
            error={errors.vramLimit}
          />
          <InputField
            label="Inference Reserve (GB)"
            value={state.vramInference}
            onChange={set('vramInference') as (v: string) => void}
            placeholder="2.0"
            hint="Reserved for inference during training."
            error={errors.vramInference}
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-text-secondary uppercase tracking-wider mb-2">
            Environment
          </label>
          <select
            value={state.environment}
            onChange={(e) => set('environment')(e.target.value)}
            className="input"
          >
            <option value="development">Development</option>
            <option value="production">Production</option>
          </select>
          <p className="text-xs text-text-muted mt-1">
            Production enables stricter error handling and disables SQL echo.
          </p>
        </div>
      </div>

      {/* GPU summary card */}
      <div className="rounded-lg p-4 space-y-2"
        style={{ background: '#0d1117', border: '1px solid #21262d' }}>
        <p className="text-xs font-medium text-text-secondary uppercase tracking-wider">GPU Budget</p>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: '#21262d' }}>
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${Math.min(100, (parseFloat(state.vramInference || '0') / parseFloat(state.vramLimit || '8')) * 100)}%`,
                background: '#238636',
              }}
            />
          </div>
          <span className="text-xs text-text-muted whitespace-nowrap">
            {state.vramInference} / {state.vramLimit} GB reserved for inference
          </span>
        </div>
      </div>
    </div>
  );
}

function StepStorage({
  state,
  set,
  errors,
}: {
  state: WizardState;
  set: (field: keyof WizardState) => (v: string | boolean) => void;
  errors: ValidationErrors;
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Storage Paths</h2>
        <p className="text-sm text-text-muted mt-1">
          Set absolute paths for data, models, and outputs. All directories are created automatically.
        </p>
      </div>

      <InfoBox>
        Use absolute paths. Tilde (<code>~</code>) is expanded automatically.
        Datasets are tracked via DVC — these directories hold runtime data, not raw annotated sets.
      </InfoBox>

      <div className="space-y-4">
        <InputField
          label="Data Root"
          value={state.dataRoot}
          onChange={set('dataRoot') as (v: string) => void}
          placeholder="/path/to/trichome-analysis/data"
          hint="Parent directory for all data subdirectories."
          error={errors.dataRoot}
          monospace
        />

        <InputField
          label="Models Directory"
          value={state.modelsDir}
          onChange={set('modelsDir') as (v: string) => void}
          placeholder="/path/to/trichome-analysis/data/models"
          hint="Where trained model weights are stored."
          monospace
        />

        <InputField
          label="Outputs Directory"
          value={state.outputsDir}
          onChange={set('outputsDir') as (v: string) => void}
          placeholder="/path/to/trichome-analysis/data/outputs"
          hint="Detection results, reports, and generated PDFs."
          monospace
        />
      </div>
    </div>
  );
}

function StepServices({
  state,
  set,
  errors,
}: {
  state: WizardState;
  set: (field: keyof WizardState) => (v: string | boolean) => void;
  errors: ValidationErrors;
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">External Services</h2>
        <p className="text-sm text-text-muted mt-1">
          Connect Label Studio, MLflow, and optionally Weights &amp; Biases.
        </p>
      </div>

      {/* Label Studio */}
      <section className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary border-b border-border pb-2">
          Label Studio (annotation)
        </p>
        <InputField
          label="URL"
          value={state.labelStudioUrl}
          onChange={set('labelStudioUrl') as (v: string) => void}
          placeholder="http://localhost:3005"
          error={errors.labelStudioUrl}
        />
        <InputField
          label="API Key"
          value={state.labelStudioKey}
          onChange={set('labelStudioKey') as (v: string) => void}
          placeholder="••••••••••••••••••••••"
          hint="Found in Label Studio → Account → Access Token. Leave empty if not using Label Studio yet."
          type="password"
        />
      </section>

      {/* MLflow */}
      <section className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary border-b border-border pb-2">
          MLflow (experiment tracking)
        </p>
        <InputField
          label="Tracking URI"
          value={state.mlflowUri}
          onChange={set('mlflowUri') as (v: string) => void}
          placeholder="http://localhost:3004"
          error={errors.mlflowUri}
        />
        <InputField
          label="Default Experiment Name"
          value={state.mlflowExperiment}
          onChange={set('mlflowExperiment') as (v: string) => void}
          placeholder="trichome-detection"
        />
      </section>

      {/* W&B (optional) */}
      <section className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary border-b border-border pb-2">
          Weights &amp; Biases (optional)
        </p>
        <Toggle
          label="Enable W&B logging"
          hint="Requires an account at wandb.ai. Disabled by default."
          value={state.useWandb}
          onChange={set('useWandb')}
        />
        {state.useWandb && (
          <>
            <InputField
              label="API Key"
              value={state.wandbKey}
              onChange={set('wandbKey') as (v: string) => void}
              placeholder="••••••••••••••••••••••"
              type="password"
            />
            <InputField
              label="Project"
              value={state.wandbProject}
              onChange={set('wandbProject') as (v: string) => void}
              placeholder="trichome-detection"
            />
          </>
        )}
      </section>
    </div>
  );
}

function StepSecurity({
  state,
  set,
  errors,
}: {
  state: WizardState;
  set: (field: keyof WizardState) => (v: string | boolean) => void;
  errors: ValidationErrors;
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Security</h2>
        <p className="text-sm text-text-muted mt-1">
          Configure the secret key and optional API authentication token.
        </p>
      </div>

      <WarnBox>
        If CTIP is accessible from the internet, <strong>set a strong API_TOKEN</strong>.
        Empty = no authentication (development mode only).
      </WarnBox>

      <div className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-text-secondary uppercase tracking-wider">
              Secret Key
            </label>
            <button
              type="button"
              onClick={() => (set('secretKey') as (v: string) => void)(generateSecretKey())}
              className="text-xs text-accent hover:text-accent-hover transition-colors"
            >
              Generate random
            </button>
          </div>
          <input
            type="password"
            value={state.secretKey}
            onChange={(e) => (set('secretKey') as (v: string) => void)(e.target.value)}
            placeholder="Leave empty to use the existing value"
            className="input font-mono text-xs"
            spellCheck={false}
          />
          <p className="text-xs text-text-muted">
            Used for session tokens. Must change from the default in production.
          </p>
        </div>

        <InputField
          label="API Token (optional)"
          value={state.apiToken}
          onChange={set('apiToken') as (v: string) => void}
          placeholder="Leave empty to disable authentication"
          hint="If set, all API requests must include: Authorization: Bearer <token>"
          type="password"
        />
      </div>

      <InfoBox>
        Authentication is bearer-token only (single user / lab environment model).
        For multi-user setups, configure nginx access controls at the network layer.
      </InfoBox>
    </div>
  );
}

function StepReview({ state }: { state: WizardState }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Review Configuration</h2>
        <p className="text-sm text-text-muted mt-1">
          Verify your settings before saving. Click &quot;Save &amp; Finish&quot; to write to{' '}
          <code className="text-accent text-xs">.env</code>.
        </p>
      </div>

      {[
        {
          title: 'Network',
          rows: [
            { label: 'PUBLIC_DOMAIN', value: state.publicDomain || '(localhost only)' },
            { label: 'PUBLIC_PORT', value: state.publicPort },
          ],
        },
        {
          title: 'Hardware',
          rows: [
            { label: 'CUDA_DEVICE', value: state.cudaDevice },
            { label: 'CUDA_VISIBLE_DEVICES', value: state.cudaVisible },
            { label: 'VRAM_LIMIT_GB', value: state.vramLimit },
            { label: 'VRAM_INFERENCE_BUDGET_GB', value: state.vramInference },
            { label: 'ENVIRONMENT', value: state.environment },
          ],
        },
        {
          title: 'Storage',
          rows: [
            { label: 'DATA_ROOT', value: state.dataRoot },
            { label: 'MODELS_DIR', value: state.modelsDir },
            { label: 'OUTPUTS_DIR', value: state.outputsDir },
          ],
        },
        {
          title: 'Services',
          rows: [
            { label: 'LABEL_STUDIO_URL', value: state.labelStudioUrl },
            { label: 'LABEL_STUDIO_API_KEY', value: state.labelStudioKey, sensitive: true },
            { label: 'MLFLOW_TRACKING_URI', value: state.mlflowUri },
            { label: 'MLFLOW_EXPERIMENT_NAME', value: state.mlflowExperiment },
            { label: 'USE_WANDB', value: state.useWandb ? 'true' : 'false' },
            ...(state.useWandb
              ? [
                  { label: 'WANDB_API_KEY', value: state.wandbKey, sensitive: true },
                  { label: 'WANDB_PROJECT', value: state.wandbProject },
                ]
              : []),
          ],
        },
        {
          title: 'Security',
          rows: [
            { label: 'SECRET_KEY', value: state.secretKey || '(unchanged)', sensitive: true },
            { label: 'API_TOKEN', value: state.apiToken || '(disabled)', sensitive: true },
          ],
        },
      ].map((section) => (
        <div key={section.title}>
          <p className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
            {section.title}
          </p>
          <div className="rounded-lg overflow-hidden"
            style={{ border: '1px solid #21262d' }}>
            {section.rows.map((row) => (
              <ReviewRow key={row.label} label={row.label} value={row.value} sensitive={row.sensitive} />
            ))}
          </div>
        </div>
      ))}

      <InfoBox>
        Settings are written atomically to <code>.env</code>. Existing keys not covered by this
        wizard (e.g. DATABASE_URL) are left untouched.
      </InfoBox>
    </div>
  );
}
