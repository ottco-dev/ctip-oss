'use client';

/** Who is logged in, for the whole UI; sends logged-out users to /login in accounts mode. */

import { createContext, useContext, useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchAuthStatus, PUBLIC_PAGES, type AuthStatus } from '@/lib/auth';
import { Sidebar } from '@/components/layout/Sidebar';
import { TopBar } from '@/components/layout/TopBar';

interface AuthContextValue {
  status: AuthStatus | undefined;
  loading: boolean;
  isAdmin: boolean;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({ status: undefined, loading: true, isAdmin: true, refresh: async () => undefined });

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ['auth-status'], queryFn: fetchAuthStatus, staleTime: 60_000, refetchOnWindowFocus: true });
  const status = q.data;
  const isPublic = PUBLIC_PAGES.includes(pathname);

  useEffect(() => {
    if (!status || status.mode !== 'accounts') return;
    if (!status.authenticated && !isPublic) {
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    } else if (status.authenticated && status.user?.must_change_password && pathname !== '/account') {
      router.replace('/account?forced=1');
    }
  }, [status, isPublic, pathname, router]);

  const isAdmin = !status || status.mode !== 'accounts' || status.user?.role === 'admin';
  const value: AuthContextValue = {
    status, loading: q.isLoading, isAdmin,
    refresh: async () => { await qc.invalidateQueries({ queryKey: ['auth-status'] }); },
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/** Login/registration pages render full screen; protected pages wait until the login state is known. */
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { status, loading } = useAuth();
  if (PUBLIC_PAGES.includes(pathname)) return <>{children}</>;
  if (loading || (status?.mode === 'accounts' && !status.authenticated)) {
    return <div className="h-screen grid place-items-center text-sm" style={{ color: 'var(--text-muted)' }}>Loading…</div>;
  }
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-y-auto p-4 lg:p-6">{children}</main>
      </div>
    </div>
  );
}
