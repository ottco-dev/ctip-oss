'use client';

import { Suspense, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Loader2, LogIn } from 'lucide-react';

function safeNext(next: string | null): string {
  return next && next.startsWith('/') && !next.startsWith('//') ? next : '/';
}

function LoginForm() {
  const params = useSearchParams();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await fetch('/api/v1/auth/login', {
        method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      if (r.ok) { window.location.assign(safeNext(params.get('next'))); return; }
      const d = await r.json().catch(() => ({}));
      setError(typeof d.detail === 'string' ? d.detail : 'Login failed.');
    } catch {
      setError('The server is not reachable. Try again in a moment.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-2xl border p-7" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
      <div className="space-y-1">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent-text">CTIP</div>
        <h1 className="text-xl font-semibold text-text-primary">Sign in</h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Cannabis Trichome Intelligence Platform</p>
      </div>
      <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <span>Username</span>
        <input id="login-username" autoFocus autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required
          className="w-full px-3 py-2 rounded-lg border text-text-primary" style={{ borderColor: 'var(--border)', background: 'var(--background)' }} />
      </label>
      <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <span>Password</span>
        <input id="login-password" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required
          className="w-full px-3 py-2 rounded-lg border text-text-primary" style={{ borderColor: 'var(--border)', background: 'var(--background)' }} />
      </label>
      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      <button type="submit" disabled={busy}
        className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium bg-accent hover:bg-accent-hover text-white disabled:opacity-60">
        {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <LogIn className="w-4 h-4" />} Sign in
      </button>
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No account? Ask an admin for an invitation link.</p>
    </form>
  );
}

export default function LoginPage() {
  return (
    <div className="min-h-screen grid place-items-center p-4" style={{ background: 'var(--background)' }}>
      <Suspense><LoginForm /></Suspense>
    </div>
  );
}
