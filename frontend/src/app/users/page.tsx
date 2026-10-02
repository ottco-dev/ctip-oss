'use client';

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Copy, KeyRound, Link2, Loader2, ShieldCheck, UserPlus, Users } from 'lucide-react';
import { useAuth } from '@/components/layout/AuthProvider';

interface UserRow {
  id: string; username: string; display_name: string; email: string; role: 'admin' | 'member';
  active: boolean; must_change_password: boolean; created_at: string; last_login: string | null;
}
interface InviteRow { id: string; role: string; note: string; created_by: string; created_at: string; expires_at: string; used_at: string | null }

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api/v1/auth${path}`, {
    credentials: 'include', ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : `HTTP ${r.status}`);
  return d as T;
}

const fmt = (iso: string | null) => (iso ? new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`).toLocaleString() : '—');

/** A secret that is shown exactly once, with a copy button. */
function OneTimeSecret({ label, value, onClose }: { label: string; value: string; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="rounded-lg border p-3 space-y-2" style={{ borderColor: 'rgba(217,119,6,.5)', background: 'rgba(217,119,6,.08)' }}>
      <p className="text-xs text-text-primary">{label} — shown only now, pass it on securely.</p>
      <div className="flex items-center gap-2">
        <code className="flex-1 break-all text-xs px-2 py-1.5 rounded bg-[var(--background)] text-text-primary">{value}</code>
        <button onClick={() => { void navigator.clipboard?.writeText(value); setCopied(true); }} className="p-1.5 rounded hover:bg-[var(--panel)]" title="Copy">
          <Copy className="w-4 h-4" />
        </button>
      </div>
      <div className="flex justify-between text-xs" style={{ color: 'var(--text-muted)' }}>
        <span>{copied ? 'Copied.' : ''}</span>
        <button onClick={onClose} className="underline">Done</button>
      </div>
    </div>
  );
}

export default function UsersPage() {
  const { status, isAdmin } = useAuth();
  const qc = useQueryClient();
  const enabled = status?.mode === 'accounts' && isAdmin;
  const users = useQuery({ queryKey: ['users'], queryFn: () => api<UserRow[]>('/users'), enabled });
  const invites = useQuery({ queryKey: ['invites'], queryFn: () => api<InviteRow[]>('/invites'), enabled });

  const [secret, setSecret] = useState<{ label: string; value: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newUser, setNewUser] = useState({ username: '', display_name: '', role: 'member' });
  const [invite, setInvite] = useState({ role: 'member', note: '', days: 7 });

  const refresh = () => { void qc.invalidateQueries({ queryKey: ['users'] }); void qc.invalidateQueries({ queryKey: ['invites'] }); };
  const onError = (e: Error) => setError(e.message);

  const create = useMutation({
    mutationFn: () => api<UserRow & { temporary_password: string }>('/users', { method: 'POST', body: JSON.stringify(newUser) }),
    onSuccess: (u) => {
      setError(null); setNewUser({ username: '', display_name: '', role: 'member' }); refresh();
      setSecret({ label: `Temporary password for ${u.username} (must be changed at first login)`, value: u.temporary_password });
    },
    onError,
  });
  const patch = useMutation({
    mutationFn: ({ id, body }: { id: string; body: Record<string, unknown> }) =>
      api<UserRow & { temporary_password?: string }>(`/users/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
    onSuccess: (u) => {
      setError(null); refresh();
      if (u.temporary_password) setSecret({ label: `New temporary password for ${u.username}`, value: u.temporary_password });
    },
    onError,
  });
  const makeInvite = useMutation({
    mutationFn: () => api<{ token: string; role: string }>('/invites', { method: 'POST', body: JSON.stringify(invite) }),
    onSuccess: (r) => {
      setError(null); refresh();
      setSecret({ label: `Invitation link (${r.role}, valid ${invite.days} days, single use)`, value: `${window.location.origin}/register?invite=${r.token}` });
    },
    onError,
  });

  if (status?.mode !== 'accounts') {
    return <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>User accounts are off on this instance. Set <code>AUTH_MODE=accounts</code> to enable them.</p>;
  }
  if (!isAdmin) return <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Only admins can manage users.</p>;

  const field = 'px-3 py-2 rounded-lg border text-sm text-text-primary';
  const fs = { borderColor: 'var(--border)', background: 'var(--background)' };
  const card = 'rounded-xl border p-5 space-y-3';
  const cs = { borderColor: 'var(--border)', background: 'var(--surface)' };
  const btn = 'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-accent hover:bg-accent-hover text-white disabled:opacity-60';
  const me = status.user?.username;

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-semibold text-text-primary flex items-center gap-2"><Users className="w-5 h-5" /> Users</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          Admins manage users, workers, settings and containers; members use the platform. Scripts can still use the API token (admin rights).
        </p>
      </div>

      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      {secret && <OneTimeSecret label={secret.label} value={secret.value} onClose={() => setSecret(null)} />}

      <div className="grid md:grid-cols-2 gap-4">
        <form className={card} style={cs} onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
          <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2"><UserPlus className="w-4 h-4" /> Create user</h2>
          <input className={`${field} w-full`} style={fs} placeholder="username" value={newUser.username} required minLength={3}
            onChange={(e) => setNewUser({ ...newUser, username: e.target.value })} />
          <input className={`${field} w-full`} style={fs} placeholder="display name (optional)" value={newUser.display_name}
            onChange={(e) => setNewUser({ ...newUser, display_name: e.target.value })} />
          <div className="flex gap-2">
            <select className={field} style={fs} value={newUser.role} onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}>
              <option value="member">member</option><option value="admin">admin</option>
            </select>
            <button className={btn} disabled={create.isPending}>{create.isPending && <Loader2 className="w-4 h-4 animate-spin" />} Create</button>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>A temporary password is generated; the user sets their own at first login.</p>
        </form>

        <form className={card} style={cs} onSubmit={(e) => { e.preventDefault(); makeInvite.mutate(); }}>
          <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2"><Link2 className="w-4 h-4" /> Invitation link</h2>
          <input className={`${field} w-full`} style={fs} placeholder="note (who is it for?)" value={invite.note}
            onChange={(e) => setInvite({ ...invite, note: e.target.value })} />
          <div className="flex gap-2 items-center">
            <select className={field} style={fs} value={invite.role} onChange={(e) => setInvite({ ...invite, role: e.target.value })}>
              <option value="member">member</option><option value="admin">admin</option>
            </select>
            <select className={field} style={fs} value={invite.days} onChange={(e) => setInvite({ ...invite, days: Number(e.target.value) })}>
              {[1, 3, 7, 14, 30].map((d) => <option key={d} value={d}>{d} day{d > 1 ? 's' : ''}</option>)}
            </select>
            <button className={btn} disabled={makeInvite.isPending}>{makeInvite.isPending && <Loader2 className="w-4 h-4 animate-spin" />} Create link</button>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>The person picks their own username and password. Each link works once.</p>
        </form>
      </div>

      <div className={card} style={cs}>
        <h2 className="text-sm font-semibold text-text-primary">Accounts ({users.data?.length ?? 0})</h2>
        {users.isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="text-left text-xs" style={{ color: 'var(--text-muted)' }}>
                <th className="py-2 pr-3">User</th><th className="pr-3">Role</th><th className="pr-3">Status</th><th className="pr-3">Last login</th><th className="pr-3">Created</th><th />
              </tr></thead>
              <tbody>
                {users.data?.map((u) => (
                  <tr key={u.id} className="border-t" style={{ borderColor: 'var(--border)', opacity: u.active ? 1 : 0.55 }}>
                    <td className="py-2 pr-3">
                      <div className="font-medium text-text-primary">{u.username}{u.username === me && <span className="text-xs" style={{ color: 'var(--text-muted)' }}> (you)</span>}</div>
                      {u.display_name && <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{u.display_name}</div>}
                    </td>
                    <td className="pr-3">
                      <select className="px-2 py-1 rounded border text-xs text-text-primary" style={fs} value={u.role} disabled={patch.isPending}
                        onChange={(e) => patch.mutate({ id: u.id, body: { role: e.target.value } })}>
                        <option value="member">member</option><option value="admin">admin</option>
                      </select>
                    </td>
                    <td className="pr-3 text-xs">
                      {u.active ? <span className="text-green-400">active</span> : <span style={{ color: 'var(--text-muted)' }}>disabled</span>}
                      {u.must_change_password && <span className="ml-1 text-amber-400">· temp. password</span>}
                    </td>
                    <td className="pr-3 text-xs" style={{ color: 'var(--text-secondary)' }}>{fmt(u.last_login)}</td>
                    <td className="pr-3 text-xs" style={{ color: 'var(--text-secondary)' }}>{fmt(u.created_at)}</td>
                    <td className="text-right whitespace-nowrap">
                      <button className="px-2 py-1 text-xs rounded hover:bg-[var(--panel)]" title="Reset password" disabled={patch.isPending}
                        onClick={() => { if (confirm(`Reset the password of ${u.username}? Their sessions are signed out.`)) patch.mutate({ id: u.id, body: { reset_password: true } }); }}>
                        <KeyRound className="w-3.5 h-3.5 inline" /> Reset
                      </button>
                      <button className="px-2 py-1 text-xs rounded hover:bg-[var(--panel)]" disabled={patch.isPending || u.username === me}
                        onClick={() => patch.mutate({ id: u.id, body: { active: !u.active } })}>
                        {u.active ? 'Disable' : 'Enable'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className={card} style={cs}>
        <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2"><ShieldCheck className="w-4 h-4" /> Invitations</h2>
        {!invites.data?.length ? <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No invitations yet.</p> : (
          <table className="w-full text-xs">
            <thead><tr className="text-left" style={{ color: 'var(--text-muted)' }}><th className="py-1 pr-3">Note</th><th className="pr-3">Role</th><th className="pr-3">By</th><th className="pr-3">Expires</th><th>State</th></tr></thead>
            <tbody>
              {invites.data.map((i) => {
                const expired = !i.used_at && new Date(i.expires_at.endsWith('Z') ? i.expires_at : `${i.expires_at}Z`) < new Date();
                return (
                  <tr key={i.id} className="border-t" style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}>
                    <td className="py-1.5 pr-3">{i.note || '—'}</td><td className="pr-3">{i.role}</td><td className="pr-3">{i.created_by}</td>
                    <td className="pr-3">{fmt(i.expires_at)}</td>
                    <td>{i.used_at ? `used ${fmt(i.used_at)}` : expired ? 'expired' : <span className="text-green-400">open</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
