'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { ChevronDown, LogOut, UserCog, UserRound, Users } from 'lucide-react';
import { logout } from '@/lib/auth';
import { useAuth } from '@/components/layout/AuthProvider';

/** Signed-in user, links to the account and (admins) user pages, and logout. Only shown in accounts mode. */
export function UserMenu() {
  const { status, isAdmin } = useAuth();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  const user = status?.user;
  if (status?.mode !== 'accounts' || !user) return null;
  const item = 'flex items-center gap-2 px-3 py-2 text-sm rounded-md hover:bg-[var(--panel)]';

  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((v) => !v)} aria-haspopup="menu" aria-expanded={open}
        className="flex items-center gap-2 px-2 py-1 rounded-md text-xs hover:bg-[var(--panel)]" style={{ color: 'var(--text-secondary)' }}>
        <UserRound className="w-4 h-4" />
        <span className="hidden sm:inline font-medium" style={{ color: 'var(--text-primary)' }}>{user.display_name || user.username}</span>
        {user.role === 'admin' && <span className="hidden sm:inline px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wide bg-accent-subtle text-accent-text">admin</span>}
        <ChevronDown className="w-3 h-3" />
      </button>
      {open && (
        <div role="menu" className="absolute right-0 mt-1 w-48 rounded-lg border p-1 z-50 shadow-lg"
          style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--text-primary)' }}>
          <div className="px-3 py-2 text-xs" style={{ color: 'var(--text-muted)' }}>@{user.username} · {user.role}</div>
          <Link href="/account" className={item} onClick={() => setOpen(false)}><UserCog className="w-4 h-4" /> Account</Link>
          {isAdmin && <Link href="/users" className={item} onClick={() => setOpen(false)}><Users className="w-4 h-4" /> Users</Link>}
          <button onClick={() => void logout()} className={`${item} w-full text-left text-red-400`}><LogOut className="w-4 h-4" /> Sign out</button>
        </div>
      )}
    </div>
  );
}
