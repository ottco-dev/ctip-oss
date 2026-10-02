'use client';

import { Suspense, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { KeyRound, Loader2 } from 'lucide-react';
import { useAuth } from '@/components/layout/AuthProvider';

function AccountForm() {
  const forced = useSearchParams().get('forced') === '1';
  const { status, refresh } = useAuth();
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [repeat, setRepeat] = useState('');
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const user = status?.user;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (next !== repeat) { setMsg({ ok: false, text: 'The new passwords do not match.' }); return; }
    setBusy(true);
    try {
      const r = await fetch('/api/v1/auth/password', {
        method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current, new: next }),
      });
      const d = await r.json().catch(() => ({}));
      if (r.ok) {
        setMsg({ ok: true, text: 'Password changed. Other sessions of this account were signed out.' });
        setCurrent(''); setNext(''); setRepeat('');
        await refresh();
      } else setMsg({ ok: false, text: typeof d.detail === 'string' ? d.detail : 'Could not change the password.' });
    } finally {
      setBusy(false);
    }
  }

  if (status?.mode !== 'accounts') {
    return <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>This instance has no user accounts (AUTH_MODE is not &quot;accounts&quot;).</p>;
  }
  const field = 'w-full px-3 py-2 rounded-lg border text-text-primary';
  const fs = { borderColor: 'var(--border)', background: 'var(--background)' };
  return (
    <div className="max-w-md space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">Your account</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Signed in as <b className="text-text-primary">{user?.username}</b>{user?.display_name ? ` (${user.display_name})` : ''} · {user?.role}
        </p>
      </div>
      {(forced || user?.must_change_password) && (
        <p className="text-sm rounded-lg border px-3 py-2" style={{ borderColor: 'rgba(217,119,6,.5)', color: 'var(--text-primary)', background: 'rgba(217,119,6,.08)' }}>
          You signed in with a temporary password. Choose your own password to continue.
        </p>
      )}
      <form onSubmit={submit} className="space-y-3 rounded-xl border p-5" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
        <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2"><KeyRound className="w-4 h-4" /> Change password</h2>
        <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}><span>Current password</span>
          <input id="acc-current" type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} required className={field} style={fs} /></label>
        <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}><span>New password (at least 10 characters)</span>
          <input id="acc-new" type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} required className={field} style={fs} /></label>
        <label className="block space-y-1 text-sm" style={{ color: 'var(--text-secondary)' }}><span>Repeat new password</span>
          <input id="acc-repeat" type="password" autoComplete="new-password" value={repeat} onChange={(e) => setRepeat(e.target.value)} required className={field} style={fs} /></label>
        {msg && <p role="status" className={msg.ok ? 'text-sm text-green-400' : 'text-sm text-red-400'}>{msg.text}</p>}
        <button type="submit" disabled={busy} className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-accent hover:bg-accent-hover text-white disabled:opacity-60">
          {busy && <Loader2 className="w-4 h-4 animate-spin" />} Save password
        </button>
      </form>
    </div>
  );
}

export default function AccountPage() {
  return <Suspense><AccountForm /></Suspense>;
}
