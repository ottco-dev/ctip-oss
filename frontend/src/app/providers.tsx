'use client';

import { QueryClient, QueryClientProvider as TanstackProvider } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { ensureSession } from '@/lib/auth';

export function QueryClientProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,       // 30s stale time
            retry: 2,
            refetchOnWindowFocus: false,
          },
        },
      }),
  );

  // token mode: ask for API_TOKEN once per browser; accounts mode: ensureSession sends to /login
  useEffect(() => {
    void ensureSession();
  }, []);

  return <TanstackProvider client={queryClient}>{children}</TanstackProvider>;
}
