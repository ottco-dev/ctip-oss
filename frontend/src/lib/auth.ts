/**
 * Login for the browser.
 *
 * accounts mode (hosted instances): username + password on /login, a server-side session in an HttpOnly cookie.
 * token mode (single user): the UI asks once for API_TOKEN and keeps it in an HttpOnly cookie.
 * off: nothing to do.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '/api/v1';

export interface AuthUser {
  username: string;
  role: 'admin' | 'member' | string;
  display_name?: string;
  must_change_password?: boolean;
}

export interface AuthStatus {
  enabled: boolean;
  authenticated: boolean;
  mode: 'off' | 'token' | 'accounts';
  user: AuthUser | null;
}

export const PUBLIC_PAGES = ['/login', '/register'];

export async function fetchAuthStatus(): Promise<AuthStatus> {
  try {
    const r = await fetch(`${API_BASE}/auth/status`, { credentials: 'include', cache: 'no-store' });
    if (!r.ok) return { enabled: false, authenticated: true, mode: 'off', user: null };
    return await r.json();
  } catch {
    return { enabled: false, authenticated: true, mode: 'off', user: null };
  }
}

export function loginUrl(): string {
  if (typeof window === 'undefined') return '/login';
  const here = window.location.pathname + window.location.search;
  return PUBLIC_PAGES.includes(window.location.pathname) ? '/login' : `/login?next=${encodeURIComponent(here)}`;
}

let pending: Promise<boolean> | null = null;

/** Called on 401s: accounts mode → go to the login page; token mode → ask for the token once. */
export function ensureSession(): Promise<boolean> {
  if (typeof window === 'undefined') return Promise.resolve(true);
  pending ??= (async () => {
    try {
      const s = await fetchAuthStatus();
      if (!s.enabled || s.authenticated) return true;
      if (s.mode === 'accounts') {
        if (!PUBLIC_PAGES.includes(window.location.pathname)) window.location.assign(loginUrl());
        return false;
      }
      for (let attempt = 0; attempt < 3; attempt++) {
        const token = window.prompt(attempt === 0 ? 'CTIP API token (API_TOKEN in .env):' : 'That token was rejected. CTIP API token:');
        if (!token) return false;
        const r = await fetch(`${API_BASE}/auth/session`, {
          method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: token.trim() }),
        });
        if (r.ok) { window.location.reload(); return true; }
      }
      return false;
    } finally {
      pending = null;
    }
  })();
  return pending;
}

export async function logout(): Promise<void> {
  await fetch(`${API_BASE}/auth/logout`, { method: 'POST', credentials: 'include' }).catch(() => undefined);
  window.location.assign('/login');
}
