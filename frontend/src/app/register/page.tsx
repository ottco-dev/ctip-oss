'use client';

import { Suspense, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Loader2, UserPlus } from 'lucide-react';

function RegisterForm() {
  const params = useSearchParams();
  const invite = params.get('invite') ?? '';
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [repeat, setRepeat] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (password !== repeat) { setError('The passwords do not match.'); return; }
    setBusy(true);
    setError(null);
    try {
      const r = await fetch('/api/v1/auth/register', {
        method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ invite, username: username.trim(), password, display_name: displayName.trim() }),
      });
      if (r.ok) { window.location.assign('/'); return; }
      const d = await r.json().catch(() => ({}));
      setError(typeof d.detail === 'string' ? d.detail : 'Registration failed - check the fields.');
    } finally {
      setBusy(false);
    }
  }

  const field = 'w-full px-3 py-2 rounded-lg border text-text-primary';
  const fs = { borderColor: 'var(--border)', background: 'var(--background)' };
  if (!invite) {
    return <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>This page needs an invitation link from an admin.</p>;
  }
  return (
    <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-2xl border p-7" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
      <div className="space-y-1">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent-text">CTIP</div>
        <h1 className="text-xl font-semibold text-text-primary">Create your account</h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>You were invited to this CTIP instance.</p>
      </div>
      <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <span>Username <span className="text-xs" style={{ color: 'var(--text-muted)' }}>(3-40 letters, digits, . _ -)</span></span>
        <input id="reg-username" autoFocus autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required className={field} style={fs} />
      </label>
      <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <span>Display name <span className="text-xs" style={{ color: 'var(--text-muted)' }}>(optional)</span></span>
        <input id="reg-display" value={displayName} onChange={(e) => setDisplayName(e.target.value)} className={field} style={fs} />
      </label>
      <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <span>Password <span className="text-xs" style={{ color: 'var(--text-muted)' }}>(at least 10 characters)</span></span>
        <input id="reg-password" type="password" autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} required className={field} style={fs} />
      </label>
      <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <span>Repeat password</span>
        <input id="reg-repeat" type="password" autoComplete="new-password" value={repeat} onChange={(e) => setRepeat(e.target.value)} required className={field} style={fs} />
      </label>
      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      <button type="submit" disabled={busy} className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium bg-accent hover:bg-accent-hover text-white disabled:opacity-60">
        {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <UserPlus className="w-4 h-4" />} Create account
      </button>
    </form>
  );
}

export default function RegisterPage() {
  return (
    <div className="min-h-screen grid place-items-center p-4" style={{ background: 'var(--background)' }}>
      <Suspense><RegisterForm /></Suspense>
    </div>
  );
}
