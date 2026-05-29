'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Cpu,
  MemoryStick,
  Zap,
  Bot,
  Database,
  Shield,
  BarChart3,
  CheckCircle,
  AlertTriangle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Save,
  RefreshCw,
  Sliders,
  Key,
  Copy,
  Trash2,
} from 'lucide-react';
import { api } from '@/lib/api';

// ── Types ────────────────────────────────────────────────────────────────────

interface DeviceInfo {
  index: number;
  name: string;
  vram_gb: number | null;
  compute_capability: string | null;
  backend: string;
}

interface ComputeInfo {
  configured_backend: string;
  resolved_device: string;
  recommended_backend: string;
  cuda_available: boolean;
  rocm_available: boolean;
  mps_available: boolean;
  cuda_device_count: number;
  torch_version: string | null;
  cuda_version: string | null;
  hip_version: string | null;
  devices: DeviceInfo[];
  gpu_semaphore_active: boolean;
}

interface PlatformSettings {
  compute_backend: string;
  cuda_device: string;
  vram_limit_gb: number;
  max_concurrent_gpu_tasks: number;
  gpu_inference_queue_depth: number;
  default_vlm_backend: string;
  vlm_min_confidence: number;
  active_vlm_provider: string;
  active_vlm_model: string;
  data_root: string;
  models_dir: string;
  uploads_dir: string;
  max_upload_size_mb: number;
  log_level: string;
  api_token_enabled: boolean;
  mlflow_tracking_uri: string;
  mlflow_experiment_name: string;
}

// ── Constants ────────────────────────────────────────────────────────────────

const COMPUTE_BACKENDS = [
  {
    id: 'auto',
    label: 'Auto-Detect',
    description: 'Best available backend detected at startup',
    icon: Zap,
    color: 'var(--accent)',
  },
  {
    id: 'cuda',
    label: 'NVIDIA CUDA',
    description: 'NVIDIA GPU via CUDA — requires PyTorch with CUDA',
    icon: Cpu,
    color: '#60a5fa',
  },
  {
    id: 'rocm',
    label: 'AMD ROCm',
    description: 'AMD GPU via ROCm/HIP — requires PyTorch ROCm build',
    icon: Cpu,
    color: '#f97316',
  },
  {
    id: 'mps',
    label: 'Apple MPS',
    description: 'Apple Silicon unified memory — macOS only',
    icon: Cpu,
    color: '#a78bfa',
  },
  {
    id: 'cpu',
    label: 'CPU-Only',
    description: 'No GPU required — slower but always available',
    icon: MemoryStick,
    color: '#6b7280',
  },
] as const;

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const;

// ── Helpers ───────────────────────────────────────────────────────────────────

function BackendBadge({ backend, available }: { backend: string; available: boolean }) {
  const color = available ? '#22c55e' : '#6b7280';
  const icon = available ? <CheckCircle size={12} /> : <XCircle size={12} />;
  return (
    <span
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        fontSize: 11, padding: '2px 7px', borderRadius: 99,
        background: available ? 'rgba(34,197,94,0.12)' : 'rgba(107,114,128,0.12)',
        color,
        border: `1px solid ${color}33`,
      }}
    >
      {icon} {available ? 'available' : 'not found'}
    </span>
  );
}

function SectionCard({
  icon: Icon, title, children,
}: {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 10,
      overflow: 'hidden',
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 10,
          padding: '14px 18px', background: 'transparent', border: 'none',
          cursor: 'pointer', color: 'var(--text)',
        }}
      >
        <Icon size={16} color="var(--accent)" />
        <span style={{ fontWeight: 600, fontSize: 14, flex: 1, textAlign: 'left' }}>{title}</span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && (
        <div style={{ padding: '0 18px 18px 18px', borderTop: '1px solid var(--border)' }}>
          {children}
        </div>
      )}
    </div>
  );
}

// ── Compute Backend Section ───────────────────────────────────────────────────

function ComputeSection({ hw, currentBackend, onSave }: {
  hw: ComputeInfo;
  currentBackend: string;
  onSave: (backend: string) => void;
}) {
  const [selected, setSelected] = useState(currentBackend);
  const [saving, setSaving] = useState(false);

  const availabilityMap: Record<string, boolean> = {
    auto: true,
    cuda: hw.cuda_available && !hw.rocm_available,
    rocm: hw.rocm_available,
    mps: hw.mps_available,
    cpu: true,
  };

  const handleSave = async () => {
    setSaving(true);
    try { await onSave(selected); } finally { setSaving(false); }
  };

  return (
    <SectionCard icon={Cpu} title="Compute Backend">
      <div style={{ marginTop: 14 }}>
        {/* Hardware summary */}
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
          gap: 10, marginBottom: 18,
        }}>
          {hw.devices.map(d => (
            <div key={d.index} style={{
              padding: '10px 14px',
              background: 'var(--bg)',
              border: '1px solid var(--border)',
              borderRadius: 8,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                {d.name}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                {d.vram_gb != null && <div>{d.vram_gb} GB VRAM</div>}
                {d.compute_capability && <div>SM {d.compute_capability}</div>}
                <div style={{ textTransform: 'uppercase', letterSpacing: 1 }}>{d.backend}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Availability chips */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: '22px' }}>Detected:</span>
          <BackendBadge backend="CUDA" available={hw.cuda_available && !hw.rocm_available} />
          <BackendBadge backend="ROCm" available={hw.rocm_available} />
          <BackendBadge backend="MPS" available={hw.mps_available} />
          {hw.torch_version && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: '22px' }}>
              PyTorch {hw.torch_version}
              {hw.cuda_version && ` · CUDA ${hw.cuda_version}`}
              {hw.hip_version && ` · ROCm ${hw.hip_version}`}
            </span>
          )}
        </div>

        {/* Backend selector */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {COMPUTE_BACKENDS.map(b => {
            const avail = availabilityMap[b.id] ?? false;
            const isActive = selected === b.id;
            const isRecommended = hw.recommended_backend === b.id;
            return (
              <label
                key={b.id}
                style={{
                  display: 'flex', alignItems: 'flex-start', gap: 12,
                  padding: '12px 14px',
                  borderRadius: 8,
                  border: `1px solid ${isActive ? b.color : 'var(--border)'}`,
                  background: isActive ? `${b.color}12` : 'var(--bg)',
                  cursor: 'pointer',
                  transition: 'all .15s',
                }}
              >
                <input
                  type="radio"
                  name="compute_backend"
                  value={b.id}
                  checked={isActive}
                  onChange={() => setSelected(b.id)}
                  style={{ marginTop: 2, accentColor: b.color }}
                />
                <b.icon size={16} color={isActive ? b.color : 'var(--text-muted)'} style={{ marginTop: 2, flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: isActive ? b.color : 'var(--text)' }}>
                      {b.label}
                    </span>
                    {isRecommended && (
                      <span style={{
                        fontSize: 10, padding: '1px 6px', borderRadius: 99,
                        background: 'rgba(74,124,69,0.18)', color: 'var(--accent)',
                        border: '1px solid var(--accent-subtle)', textTransform: 'uppercase', letterSpacing: 1,
                      }}>recommended</span>
                    )}
                    <BackendBadge backend="" available={avail} />
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                    {b.description}
                  </div>
                </div>
              </label>
            );
          })}
        </div>

        {/* AMD ROCm install hint */}
        {selected === 'rocm' && !hw.rocm_available && (
          <div style={{
            marginTop: 12, padding: '10px 14px',
            background: 'rgba(249,115,22,0.08)',
            border: '1px solid rgba(249,115,22,0.3)',
            borderRadius: 8, fontSize: 12, color: '#f97316',
          }}>
            <strong>ROCm setup required:</strong> Install PyTorch with ROCm support:<br/>
            <code style={{ fontFamily: 'monospace', display: 'block', marginTop: 6 }}>
              pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
            </code>
          </div>
        )}

        {/* Current resolved device */}
        <div style={{ marginTop: 14, fontSize: 12, color: 'var(--text-muted)' }}>
          Currently resolved to: <code style={{
            background: 'var(--bg)', border: '1px solid var(--border)',
            borderRadius: 4, padding: '1px 6px', fontFamily: 'monospace',
          }}>{hw.resolved_device}</code>
          {hw.gpu_semaphore_active && (
            <span style={{ marginLeft: 10, color: '#f59e0b' }}>⚠ GPU semaphore active — job running</span>
          )}
        </div>

        <button
          onClick={handleSave}
          disabled={saving || selected === currentBackend}
          style={{
            marginTop: 16, display: 'flex', alignItems: 'center', gap: 6,
            padding: '8px 16px', borderRadius: 7,
            background: selected !== currentBackend ? 'var(--accent)' : 'var(--border)',
            color: selected !== currentBackend ? '#fff' : 'var(--text-muted)',
            border: 'none', cursor: selected !== currentBackend ? 'pointer' : 'not-allowed',
            fontSize: 13, fontWeight: 600,
            opacity: saving ? 0.7 : 1,
          }}
        >
          {saving ? <RefreshCw size={14} className="animate-spin" /> : <Save size={14} />}
          {saving ? 'Saving…' : 'Save Compute Backend'}
        </button>
      </div>
    </SectionCard>
  );
}

// ── VLM Section ───────────────────────────────────────────────────────────────

function VlmSection({ settings, onSave }: {
  settings: PlatformSettings;
  onSave: (patch: Partial<PlatformSettings>) => Promise<void>;
}) {
  const [backend, setBackend] = useState(settings.default_vlm_backend);
  const [threshold, setThreshold] = useState(String(settings.vlm_min_confidence));
  const [saving, setSaving] = useState(false);

  const VLM_BACKENDS = [
    { id: 'moondream', label: 'Moondream-2B', kind: 'local', note: 'FP16, ~4.2 GB VRAM' },
    { id: 'florence2', label: 'Florence-2', kind: 'local', note: '~3 GB VRAM' },
    { id: 'qwen2vl', label: 'Qwen2-VL', kind: 'local', note: '~6 GB VRAM' },
    { id: 'openai', label: 'OpenAI GPT-4o', kind: 'remote', note: 'API key required' },
    { id: 'anthropic', label: 'Anthropic Claude', kind: 'remote', note: 'API key required' },
    { id: 'google', label: 'Google Gemini', kind: 'remote', note: 'Free tier available' },
    { id: 'groq', label: 'Groq', kind: 'remote', note: 'Free tier available' },
  ];

  const handleSave = async () => {
    setSaving(true);
    const thresholdVal = parseFloat(threshold);
    if (isNaN(thresholdVal) || thresholdVal < 0 || thresholdVal > 1) {
      alert('Confidence threshold must be between 0 and 1');
      setSaving(false);
      return;
    }
    try {
      await onSave({ default_vlm_backend: backend, vlm_min_confidence: thresholdVal });
    } finally { setSaving(false); }
  };

  return (
    <SectionCard icon={Bot} title="VLM Auto-Labeling">
      <div style={{ marginTop: 14 }}>
        <label style={{ display: 'block', marginBottom: 8, fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
          Default VLM Backend
        </label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 18 }}>
          {VLM_BACKENDS.map(b => (
            <label key={b.id} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '9px 12px', borderRadius: 7,
              border: `1px solid ${backend === b.id ? 'var(--accent)' : 'var(--border)'}`,
              background: backend === b.id ? 'var(--accent-subtle)' : 'var(--bg)',
              cursor: 'pointer',
            }}>
              <input type="radio" name="vlm_backend" value={b.id}
                checked={backend === b.id} onChange={() => setBackend(b.id)}
                style={{ accentColor: 'var(--accent)' }}
              />
              <span style={{ flex: 1, fontSize: 13, color: 'var(--text)' }}>{b.label}</span>
              <span style={{
                fontSize: 10, padding: '1px 6px', borderRadius: 99,
                background: b.kind === 'local' ? 'rgba(74,124,69,0.15)' : 'rgba(96,165,250,0.15)',
                color: b.kind === 'local' ? 'var(--accent)' : '#60a5fa',
                border: `1px solid ${b.kind === 'local' ? 'var(--accent-subtle)' : 'rgba(96,165,250,0.3)'}`,
              }}>{b.kind}</span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{b.note}</span>
            </label>
          ))}
        </div>

        <label style={{ display: 'block', marginBottom: 6, fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
          Minimum Confidence Threshold
          <span style={{ fontWeight: 400, color: 'var(--text-muted)', marginLeft: 8, fontSize: 12 }}>
            Labels below this are flagged for mandatory human review
          </span>
        </label>
        <input
          type="number" min="0" max="1" step="0.01"
          value={threshold}
          onChange={e => setThreshold(e.target.value)}
          style={{
            width: 100, padding: '6px 10px', borderRadius: 6,
            background: 'var(--bg)', border: '1px solid var(--border)',
            color: 'var(--text)', fontSize: 13,
          }}
        />
        <span style={{ marginLeft: 10, fontSize: 12, color: 'var(--text-muted)' }}>
          Current: {settings.vlm_min_confidence} — Active provider: <strong>{settings.active_vlm_provider}</strong>
          {settings.active_vlm_model && ` (${settings.active_vlm_model})`}
        </span>

        <div style={{ marginTop: 16 }}>
          <button onClick={handleSave} disabled={saving} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '8px 16px', borderRadius: 7,
            background: 'var(--accent)', color: '#fff',
            border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            opacity: saving ? 0.7 : 1,
          }}>
            {saving ? <RefreshCw size={14} className="animate-spin" /> : <Save size={14} />}
            {saving ? 'Saving…' : 'Save VLM Settings'}
          </button>
        </div>
      </div>
    </SectionCard>
  );
}

// ── Performance Section ───────────────────────────────────────────────────────

function PerformanceSection({ settings, onSave }: {
  settings: PlatformSettings;
  onSave: (patch: Partial<PlatformSettings>) => Promise<void>;
}) {
  const [vram, setVram] = useState(String(settings.vram_limit_gb));
  const [queueDepth, setQueueDepth] = useState(String(settings.gpu_inference_queue_depth));
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave({
        vram_limit_gb: parseFloat(vram),
        gpu_inference_queue_depth: parseInt(queueDepth, 10),
      });
    } finally { setSaving(false); }
  };

  return (
    <SectionCard icon={Sliders} title="Performance & GPU Guard">
      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', display: 'block', marginBottom: 6 }}>
            VRAM Limit (GB)
          </label>
          <input type="number" min="1" max="80" step="0.5" value={vram}
            onChange={e => setVram(e.target.value)}
            style={{ width: 100, padding: '6px 10px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: 13 }}
          />
          <span style={{ marginLeft: 10, fontSize: 12, color: 'var(--text-muted)' }}>
            Used by GPU guard middleware to limit concurrent VRAM usage
          </span>
        </div>

        <div>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', display: 'block', marginBottom: 6 }}>
            GPU Inference Queue Depth
          </label>
          <input type="number" min="0" max="32" step="1" value={queueDepth}
            onChange={e => setQueueDepth(e.target.value)}
            style={{ width: 100, padding: '6px 10px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: 13 }}
          />
          <span style={{ marginLeft: 10, fontSize: 12, color: 'var(--text-muted)' }}>
            0 = fail-fast (HTTP 429 immediately) · 1+ = queue requests
          </span>
        </div>

        <div>
          <button onClick={handleSave} disabled={saving} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '8px 16px', borderRadius: 7, background: 'var(--accent)', color: '#fff',
            border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            opacity: saving ? 0.7 : 1,
          }}>
            {saving ? <RefreshCw size={14} className="animate-spin" /> : <Save size={14} />}
            {saving ? 'Saving…' : 'Save Performance Settings'}
          </button>
        </div>
      </div>
    </SectionCard>
  );
}

// ── Logging Section ───────────────────────────────────────────────────────────

function LoggingSection({ settings, onSave }: {
  settings: PlatformSettings;
  onSave: (patch: Partial<PlatformSettings>) => Promise<void>;
}) {
  const [level, setLevel] = useState(settings.log_level);
  const [mlflowUri, setMlflowUri] = useState(settings.mlflow_tracking_uri);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try { await onSave({ log_level: level, mlflow_tracking_uri: mlflowUri }); }
    finally { setSaving(false); }
  };

  return (
    <SectionCard icon={BarChart3} title="Logging & MLflow">
      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', display: 'block', marginBottom: 6 }}>
            Log Level
          </label>
          <div style={{ display: 'flex', gap: 8 }}>
            {LOG_LEVELS.map(l => (
              <button key={l} onClick={() => setLevel(l)} style={{
                padding: '5px 12px', borderRadius: 6, fontSize: 12, fontWeight: 600,
                border: `1px solid ${level === l ? 'var(--accent)' : 'var(--border)'}`,
                background: level === l ? 'var(--accent-subtle)' : 'var(--bg)',
                color: level === l ? 'var(--accent)' : 'var(--text-muted)',
                cursor: 'pointer',
              }}>{l}</button>
            ))}
          </div>
        </div>

        <div>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', display: 'block', marginBottom: 6 }}>
            MLflow Tracking URI
          </label>
          <input type="text" value={mlflowUri} onChange={e => setMlflowUri(e.target.value)}
            style={{ width: 320, padding: '6px 10px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)', fontSize: 13 }}
          />
        </div>

        <button onClick={handleSave} disabled={saving} style={{
          alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 6,
          padding: '8px 16px', borderRadius: 7, background: 'var(--accent)', color: '#fff',
          border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          opacity: saving ? 0.7 : 1,
        }}>
          {saving ? <RefreshCw size={14} className="animate-spin" /> : <Save size={14} />}
          {saving ? 'Saving…' : 'Save Logging Settings'}
        </button>
      </div>
    </SectionCard>
  );
}

// ── Storage Section ───────────────────────────────────────────────────────────

function StorageSection({ settings }: { settings: PlatformSettings }) {
  return (
    <SectionCard icon={Database} title="Storage Paths">
      <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {[
          { label: 'Data Root', value: settings.data_root },
          { label: 'Models Dir', value: settings.models_dir },
          { label: 'Uploads Dir', value: settings.uploads_dir },
          { label: 'Max Upload Size', value: `${settings.max_upload_size_mb} MB` },
        ].map(item => (
          <div key={item.label} style={{
            padding: '10px 14px',
            background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8,
          }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{item.label}</div>
            <code style={{ fontSize: 12, color: 'var(--text)', fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {item.value}
            </code>
          </div>
        ))}
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 10, marginBottom: 0 }}>
        Paths are configured via .env and require a server restart to change.
      </p>
    </SectionCard>
  );
}

// ── Security Section ──────────────────────────────────────────────────────────

function SecuritySection({ settings }: { settings: PlatformSettings }) {
  return (
    <SectionCard icon={Shield} title="Security">
      <div style={{ marginTop: 14 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
          background: settings.api_token_enabled ? 'rgba(34,197,94,0.08)' : 'rgba(245,158,11,0.08)',
          border: `1px solid ${settings.api_token_enabled ? 'rgba(34,197,94,0.3)' : 'rgba(245,158,11,0.3)'}`,
          borderRadius: 8,
        }}>
          {settings.api_token_enabled
            ? <CheckCircle size={16} color="#22c55e" />
            : <AlertTriangle size={16} color="#f59e0b" />
          }
          <span style={{ fontSize: 13, color: 'var(--text)' }}>
            {settings.api_token_enabled
              ? 'API token authentication enabled'
              : 'Authentication disabled — development mode'}
          </span>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 10, marginBottom: 0 }}>
          Set <code style={{ fontFamily: 'monospace', background: 'var(--bg)', padding: '1px 5px', borderRadius: 3 }}>API_TOKEN</code> in your .env file and restart the server to enable authentication.
        </p>
      </div>
    </SectionCard>
  );
}

// ── API Security Section ──────────────────────────────────────────────────────

interface TokenStatus {
  enabled: boolean;
  token_preview: string | null;
  created_hint: string | null;
}

function ApiSecuritySection() {
  const qc = useQueryClient();
  const [newToken, setNewToken] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  const tokenQuery = useQuery<TokenStatus>({
    queryKey: ['system', 'token-status'],
    queryFn: () => api.get('/system/token/status').then(r => r.data),
    staleTime: 10_000,
  });

  const generateMutation = useMutation({
    mutationFn: () => api.post('/system/token/generate').then(r => r.data),
    onSuccess: (data: { token: string; warning: string }) => {
      setNewToken(data.token);
      setCopied(false);
      qc.invalidateQueries({ queryKey: ['system', 'token-status'] });
    },
  });

  const clearMutation = useMutation({
    mutationFn: () => api.post('/system/token/clear').then(r => r.data),
    onSuccess: () => {
      setNewToken(null);
      setCopied(false);
      setConfirmClear(false);
      qc.invalidateQueries({ queryKey: ['system', 'token-status'] });
    },
  });

  const handleCopy = async () => {
    if (!newToken) return;
    try {
      await navigator.clipboard.writeText(newToken);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      // fallback for non-secure context
    }
  };

  const status = tokenQuery.data;

  return (
    <SectionCard icon={Key} title="API Security">
      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>

        {/* Auth status badge */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
          background: status?.enabled ? 'rgba(34,197,94,0.08)' : 'rgba(245,158,11,0.08)',
          border: `1px solid ${status?.enabled ? 'rgba(34,197,94,0.3)' : 'rgba(245,158,11,0.3)'}`,
          borderRadius: 8,
        }}>
          {status?.enabled
            ? <CheckCircle size={16} color="#22c55e" />
            : <AlertTriangle size={16} color="#f59e0b" />
          }
          <span style={{ fontSize: 13, color: 'var(--text)', flex: 1 }}>
            {status?.enabled
              ? 'Token authentication enabled'
              : 'Authentication disabled — development mode'}
          </span>
          {status?.enabled && (
            <span style={{
              fontSize: 10, padding: '2px 7px', borderRadius: 99,
              background: 'rgba(34,197,94,0.12)', color: '#22c55e',
              border: '1px solid rgba(34,197,94,0.3)',
              fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase',
            }}>
              active
            </span>
          )}
        </div>

        {/* Masked token preview */}
        {status?.enabled && status.token_preview && (
          <div style={{
            padding: '10px 14px',
            background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8,
          }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
              Current token (masked)
            </div>
            <code style={{
              fontFamily: 'monospace', fontSize: 13, color: 'var(--text)',
              letterSpacing: 1,
            }}>
              {status.token_preview}
            </code>
          </div>
        )}

        {/* Newly generated token — shown only once */}
        {newToken && (
          <div style={{
            padding: '12px 14px',
            background: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.4)',
            borderRadius: 8,
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              marginBottom: 8, color: '#f59e0b', fontSize: 12, fontWeight: 600,
            }}>
              <AlertTriangle size={14} />
              Copy now — this is shown only once
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <code style={{
                flex: 1, fontFamily: 'monospace', fontSize: 12,
                padding: '7px 10px', borderRadius: 6,
                background: 'var(--bg)', border: '1px solid var(--border)',
                color: 'var(--text)', wordBreak: 'break-all',
                userSelect: 'all',
              }}>
                {newToken}
              </code>
              <button
                onClick={handleCopy}
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '7px 12px', borderRadius: 6, flexShrink: 0,
                  background: copied ? 'rgba(34,197,94,0.15)' : 'var(--border)',
                  color: copied ? '#22c55e' : 'var(--text)',
                  border: `1px solid ${copied ? 'rgba(34,197,94,0.3)' : 'var(--border)'}`,
                  cursor: 'pointer', fontSize: 12, fontWeight: 600,
                }}
              >
                <Copy size={13} />
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
          </div>
        )}

        {/* Actions */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <button
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '8px 16px', borderRadius: 7,
              background: 'var(--accent)', color: '#fff',
              border: 'none', cursor: generateMutation.isPending ? 'not-allowed' : 'pointer',
              fontSize: 13, fontWeight: 600,
              opacity: generateMutation.isPending ? 0.7 : 1,
            }}
          >
            {generateMutation.isPending
              ? <RefreshCw size={14} className="animate-spin" />
              : <Key size={14} />
            }
            {generateMutation.isPending ? 'Generating…' : 'Generate new token'}
          </button>

          {status?.enabled && !confirmClear && (
            <button
              onClick={() => setConfirmClear(true)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '8px 16px', borderRadius: 7,
                background: 'transparent', color: '#ef4444',
                border: '1px solid rgba(239,68,68,0.4)',
                cursor: 'pointer', fontSize: 13, fontWeight: 600,
              }}
            >
              <Trash2 size={14} />
              Remove token
            </button>
          )}

          {confirmClear && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 12, color: '#ef4444' }}>
                Confirm disable auth?
              </span>
              <button
                onClick={() => clearMutation.mutate()}
                disabled={clearMutation.isPending}
                style={{
                  padding: '7px 14px', borderRadius: 7, fontSize: 12, fontWeight: 700,
                  background: '#ef4444', color: '#fff', border: 'none',
                  cursor: clearMutation.isPending ? 'not-allowed' : 'pointer',
                  opacity: clearMutation.isPending ? 0.7 : 1,
                }}
              >
                {clearMutation.isPending ? 'Removing…' : 'Yes, remove'}
              </button>
              <button
                onClick={() => setConfirmClear(false)}
                style={{
                  padding: '7px 14px', borderRadius: 7, fontSize: 12,
                  background: 'var(--border)', color: 'var(--text)', border: 'none',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>

        {/* Error states */}
        {generateMutation.isError && (
          <div style={{
            padding: '8px 12px', borderRadius: 7, fontSize: 12,
            background: 'rgba(239,68,68,0.1)', color: '#ef4444',
            border: '1px solid rgba(239,68,68,0.3)',
          }}>
            Generate failed: {(generateMutation.error as Error)?.message}
          </div>
        )}
        {clearMutation.isError && (
          <div style={{
            padding: '8px 12px', borderRadius: 7, fontSize: 12,
            background: 'rgba(239,68,68,0.1)', color: '#ef4444',
            border: '1px solid rgba(239,68,68,0.3)',
          }}>
            Clear failed: {(clearMutation.error as Error)?.message}
          </div>
        )}

        {/* Informational note */}
        <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0 }}>
          Token is written to{' '}
          <code style={{ fontFamily: 'monospace', background: 'var(--bg)', padding: '1px 5px', borderRadius: 3 }}>
            .env
          </code>{' '}
          as{' '}
          <code style={{ fontFamily: 'monospace', background: 'var(--bg)', padding: '1px 5px', borderRadius: 3 }}>
            API_TOKEN
          </code>
          {' '}and takes effect immediately (no restart required). All API requests must include{' '}
          <code style={{ fontFamily: 'monospace', background: 'var(--bg)', padding: '1px 5px', borderRadius: 3 }}>
            Authorization: Bearer &lt;token&gt;
          </code>{' '}
          or{' '}
          <code style={{ fontFamily: 'monospace', background: 'var(--bg)', padding: '1px 5px', borderRadius: 3 }}>
            X-API-Key: &lt;token&gt;
          </code>
          .
        </p>
      </div>
    </SectionCard>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const qc = useQueryClient();

  const hwQuery = useQuery<ComputeInfo>({
    queryKey: ['settings', 'compute'],
    queryFn: () => api.get('/settings/compute').then(r => r.data),
    staleTime: 30_000,
  });

  const settingsQuery = useQuery<PlatformSettings>({
    queryKey: ['settings'],
    queryFn: () => api.get('/settings').then(r => r.data),
    staleTime: 30_000,
  });

  const setComputeMutation = useMutation({
    mutationFn: (backend: string) =>
      api.post('/settings/compute', { backend }).then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['settings'] });
      qc.invalidateQueries({ queryKey: ['settings', 'compute'] });
    },
  });

  const patchMutation = useMutation({
    mutationFn: (patch: Record<string, unknown>) =>
      api.patch('/settings', patch).then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['settings'] });
    },
  });

  const loading = hwQuery.isLoading || settingsQuery.isLoading;
  const hw = hwQuery.data;
  const s = settingsQuery.data;

  return (
    <div style={{ padding: '28px 32px', maxWidth: 840 }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text)', margin: 0 }}>
          Platform Settings
        </h1>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4, marginBottom: 0 }}>
          All changes are persisted to .env and take effect immediately without restart.
        </p>
      </div>

      {loading && (
        <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading settings…</div>
      )}

      {hw && s && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <ComputeSection
            hw={hw}
            currentBackend={s.compute_backend}
            onSave={backend => setComputeMutation.mutateAsync(backend)}
          />

          {(setComputeMutation.isError || setComputeMutation.isSuccess) && (
            <div style={{
              padding: '8px 14px', borderRadius: 8, fontSize: 13,
              background: setComputeMutation.isError ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)',
              color: setComputeMutation.isError ? '#ef4444' : '#22c55e',
              border: `1px solid ${setComputeMutation.isError ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`,
            }}>
              {setComputeMutation.isError
                ? `Error: ${(setComputeMutation.error as Error)?.message}`
                : 'Compute backend saved successfully'}
            </div>
          )}

          <VlmSection
            settings={s}
            onSave={patch => patchMutation.mutateAsync(patch as Record<string, unknown>)}
          />

          <PerformanceSection
            settings={s}
            onSave={patch => patchMutation.mutateAsync(patch as Record<string, unknown>)}
          />

          <LoggingSection
            settings={s}
            onSave={patch => patchMutation.mutateAsync(patch as Record<string, unknown>)}
          />

          <StorageSection settings={s} />

          <SecuritySection settings={s} />

          <ApiSecuritySection />

          {patchMutation.isSuccess && (
            <div style={{
              padding: '8px 14px', borderRadius: 8, fontSize: 13,
              background: 'rgba(34,197,94,0.1)', color: '#22c55e',
              border: '1px solid rgba(34,197,94,0.3)',
            }}>
              Settings saved successfully
            </div>
          )}
          {patchMutation.isError && (
            <div style={{
              padding: '8px 14px', borderRadius: 8, fontSize: 13,
              background: 'rgba(239,68,68,0.1)', color: '#ef4444',
              border: '1px solid rgba(239,68,68,0.3)',
            }}>
              Error: {(patchMutation.error as Error)?.message}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
