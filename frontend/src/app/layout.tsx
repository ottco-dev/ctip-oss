import type { Metadata } from 'next';
import { QueryClientProvider } from './providers';
import { Sidebar } from '@/components/layout/Sidebar';
import { TopBar } from '@/components/layout/TopBar';
import '@/styles/globals.css';

export const metadata: Metadata = {
  title: 'TrichomeLab — Cannabis Trichome Analysis',
  description:
    'Research-grade cannabis trichome analysis platform: detection, segmentation, maturity analysis, VLM labeling, and training.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-text-primary">
        <QueryClientProvider>
          <div className="flex h-screen overflow-hidden">
            {/* Sidebar */}
            <Sidebar />

            {/* Main content */}
            <div className="flex flex-col flex-1 overflow-hidden">
              <TopBar />
              <main className="flex-1 overflow-y-auto p-4 lg:p-6">
                {children}
              </main>
            </div>
          </div>
        </QueryClientProvider>
      </body>
    </html>
  );
}
