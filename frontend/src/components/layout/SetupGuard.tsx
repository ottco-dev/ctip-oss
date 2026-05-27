'use client';

/**
 * SetupGuard — redirects to /setup on first launch.
 *
 * Checks GET /api/v1/setup/status once per browser session.
 * If SETUP_COMPLETED is false and we're not already on /setup → redirect.
 * API failure or timeout is treated as "setup done" to avoid blocking the UI.
 */

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { api } from '@/lib/api';

const SESSION_KEY = 'ctip-setup-checked';

export function SetupGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // Already on setup page — nothing to do
    if (pathname === '/setup') return;

    // Already checked this session — skip API call
    if (sessionStorage.getItem(SESSION_KEY)) return;

    let cancelled = false;

    (async () => {
      try {
        const res = await api.get('/setup/status', { timeout: 3000 });
        if (!cancelled && res.data.completed === false) {
          router.replace('/setup');
        }
      } catch {
        // Backend unreachable (still starting up, Docker not running, etc.)
        // Fail open — don't block the UI.
      } finally {
        if (!cancelled) {
          // Mark as checked so we don't redirect on every navigation
          sessionStorage.setItem(SESSION_KEY, '1');
        }
      }
    })();

    return () => { cancelled = true; };
  }, [pathname, router]);

  return <>{children}</>;
}
