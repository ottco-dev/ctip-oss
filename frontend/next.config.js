/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // API proxy: avoid CORS in development
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/ws/:path*',
        destination: 'http://localhost:8000/ws/:path*',
      },
    ];
  },

  // Allow images from localhost (backend serves microscopy images)
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
    ],
  },

  // Experimental features
  experimental: {
    optimizePackageImports: ['lucide-react', 'recharts'],
  },
};

module.exports = nextConfig;
