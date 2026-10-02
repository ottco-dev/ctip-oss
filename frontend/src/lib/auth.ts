/**
 * API token session for the browser.
 *
 * When the backend runs with API_TOKEN set, every request needs the token. The UI asks for it once, posts it to
 * /api/v1/auth/session and the backend stores it in an HttpOnly cookie — which the browser then sends with every
 * fetch, <img> request and WebSocket upgrade. Without API_TOKEN nothing happens.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '/api/v1';

let pending: Promise<boolean> | null = null;

async function status(): Promise<{ enabled: boolean; authenticated: boolean }> {
  const r = await fetch(`${API_BASE}/auth/status`, { credentials: 'include', cache: 'no-store' });
  if (!r.ok) return { enabled: false, authenticated: true };
  return r.json();
}

/** Make sure this browser carries a valid token; asks the user if needed. Resolves true when authenticated. */
export function ensureSession(): Promise<boolean> {
  if (typeof window === 'undefined') return Promise.resolve(true);
  pending ??= (async () => {
    try {
      const s = await status();
      if (!s.enabled || s.authenticated) return true;
      for (let attempt = 0; attempt < 3; attempt++) {
        const token = window.prompt(
          attempt === 0
            ? 'CTIP API token (API_TOKEN in .env):'
            : 'That token was rejected. CTIP API token:',
        );
        if (!token) return false;
        const r = await fetch(`${API_BASE}/auth/session`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: token.trim() }),
        });
        if (r.ok) {
          window.location.reload();   // reconnect WebSockets and reload data with the cookie
          return true;
        }
      }
      return false;
    } catch {
      return true;                     // backend unreachable: let the normal error handling show it
    } finally {
      pending = null;
    }
  })();
  return pending;
}
