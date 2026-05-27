import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark mode color palette (GitHub-inspired dark)
        background: '#080b10',
        surface: '#0d1117',
        panel: '#161b22',
        border: '#21262d',
        'border-muted': '#30363d',

        // Accent colors
        accent: {
          DEFAULT: '#238636',   // GitHub green
          hover: '#2ea043',
          muted: '#1a3a24',
        },

        // Text colors
        text: {
          primary: '#e6edf3',
          secondary: '#8b949e',
          muted: '#484f58',
        },

        // Trichome maturity colors (matches Python MATURITY_COLORS)
        maturity: {
          clear: '#60a5fa',       // blue-400
          cloudy: '#f9fafb',      // gray-50
          amber: '#f59e0b',       // amber-500
          'cloudy-amber': '#d97706',  // amber-600
          degraded: '#6b7280',    // gray-500
          unknown: '#4b5563',     // gray-600
        },

        // Trichome type colors
        trichome: {
          stalked: '#22d3ee',     // cyan-400
          sessile: '#34d399',     // emerald-400
          bulbous: '#a78bfa',     // violet-400
          'non-glandular': '#fb923c', // orange-400
        },

        // Status colors
        status: {
          success: '#22c55e',
          warning: '#eab308',
          error: '#ef4444',
          info: '#3b82f6',
          pending: '#8b949e',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'metric': ['2rem', { lineHeight: '1.2', fontWeight: '700' }],
        'metric-sm': ['1.5rem', { lineHeight: '1.2', fontWeight: '600' }],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.2s ease-in-out',
        'slide-in': 'slideIn 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
  ],
};

export default config;
