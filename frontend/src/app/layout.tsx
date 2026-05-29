import type { Metadata } from 'next';
import { QueryClientProvider } from './providers';
import { Sidebar } from '@/components/layout/Sidebar';
import { TopBar } from '@/components/layout/TopBar';
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
            <SetupGuard>
              <div className="flex h-screen overflow-hidden">
                <Sidebar />
                <div className="flex flex-col flex-1 overflow-hidden">
                  <TopBar />
                  <main className="flex-1 overflow-y-auto p-4 lg:p-6">
                    {children}
                  </main>
                </div>
              </div>
            </SetupGuard>
          </ThemeProvider>
        </QueryClientProvider>
      </body>
    </html>
  );
}
