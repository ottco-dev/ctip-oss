/** @type {import('next').NextConfig} */

// Backend URL for server-side rewrites (Next.js dev proxy).
// In production the nginx reverse proxy handles /api/* and /ws/* instead.
// Override via env var if your backend is on a different host/port.
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

const nextConfig = {
  reactStrictMode: true,

  // ── ESM packages that Next.js must transpile ──────────────────────────────
  transpilePackages: ['@xyflow/react', '@xyflow/system'],

  // ── Dev proxy ────────────────────────────────────────────────────────────
  // Rewrites run server-side (Node.js), so they always reach `localhost:8000`
  // even when the browser is coming from another machine.
  // Nginx handles the same routing in production.
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: '/ws/:path*',
        destination: `${BACKEND_URL}/ws/:path*`,
      },
    ];
  },

  // ── Allow images from any host in the local subnet ───────────────────────
  // Next.js <Image> optimisation requires explicit hostname allowlist.
  images: {
    remotePatterns: [
      // Local backend (dev)
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
      // Backend accessed from another machine via nginx
      {
        protocol: 'http',
        hostname: '**',          // any LAN IP
        pathname: '/api/**',
      },
    ],
  },

  // ── Experimental ─────────────────────────────────────────────────────────
  experimental: {
    optimizePackageImports: ['lucide-react', 'recharts'],
  },
};

module.exports = nextConfig;
