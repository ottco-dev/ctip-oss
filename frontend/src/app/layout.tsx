import type { Metadata } from 'next';
import { QueryClientProvider } from './providers';
import { AppShell, AuthProvider } from '@/components/layout/AuthProvider';
import { SetupGuard } from '@/components/layout/SetupGuard';
import { ThemeProvider, themeScript } from '@/components/layout/ThemeProvider';
import '@/styles/globals.css';

export const metadata: Metadata = {
  title: 'CTIP — Cannabis Trichome Intelligence Platform',
  description:
    'CTIP: professional cannabis trichome analysis platform — detection, segmentation, maturity analysis, VLM labeling, and model training.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Inject theme before first paint to prevent flash */}
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="bg-background text-text-primary">
        <QueryClientProvider>
          <ThemeProvider>
            <AuthProvider>
              <SetupGuard>
                <AppShell>{children}</AppShell>
              </SetupGuard>
            </AuthProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </body>
    </html>
  );
}
