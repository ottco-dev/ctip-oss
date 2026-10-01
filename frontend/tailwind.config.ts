import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'selector',
  theme: {
    extend: {
      colors: {
        // CSS-variable driven — adapts to dark/light theme
        background: 'rgb(var(--background-rgb) / <alpha-value>)',
        surface: 'rgb(var(--surface-rgb) / <alpha-value>)',
        panel: 'rgb(var(--panel-rgb) / <alpha-value>)',
        border: 'rgb(var(--border-rgb) / <alpha-value>)',
        'border-muted': 'rgb(var(--border-muted-rgb) / <alpha-value>)',

        // Accent colors
        accent: {
          DEFAULT: 'rgb(var(--accent-rgb) / <alpha-value>)',
          hover: 'rgb(var(--accent-hover-rgb) / <alpha-value>)',
          subtle: 'var(--accent-subtle)',
          text: 'rgb(var(--accent-text-rgb) / <alpha-value>)',
        },

        // Text colors
        text: {
          primary: 'rgb(var(--text-primary-rgb) / <alpha-value>)',
          secondary: 'rgb(var(--text-secondary-rgb) / <alpha-value>)',
          muted: 'rgb(var(--text-muted-rgb) / <alpha-value>)',
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
