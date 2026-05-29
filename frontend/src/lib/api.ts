/**
 * API client — axios instance with base URL and error handling.
 *
 * URL strategy (works from any host/IP, not just localhost):
 *
 *   REST:  relative "/api/v1" → resolved against the page's own origin.
 *          • Via nginx on :3001 → nginx proxies /api/v1/ → backend :8000
 *          • Via Next.js on :3000 → next.config.js rewrites /api/* → backend :8000
 *          • Via another PC's IP → same rules, just different origin host
 *
 *   WS:    derived from window.location so the WS connects to the same
 *          host/port the user opened — nginx/Next.js rewrites handle the rest.
 *          Falls back to localhost:3001 during SSR/build.
 *
 * Override for special deployments:
 *   NEXT_PUBLIC_API_URL=https://api.my-domain.com/api/v1
 *   NEXT_PUBLIC_WS_URL=wss://api.my-domain.com
 */

import axios, { AxiosError, type AxiosInstance } from 'axios';

// Relative URL — works from any host.  Override via env var for remote backends.
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '/api/v1';

// Derive WS origin from the browser's current location at runtime.
// During SSR (no window) we fall back to a harmless placeholder.
function _wsBase(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window === 'undefined') return 'ws://localhost:3001';
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}`;
}

export const api: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: add timestamp for debugging
api.interceptors.request.use((config) => {
  config.headers['X-Request-Time'] = Date.now().toString();
  return config;
});

// Response interceptor: normalize errors
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const message =
      (error.response?.data as { detail?: string })?.detail ||
      error.message ||
      'Unknown error';
    return Promise.reject(new Error(message));
  },
);

export const wsUrl = (path: string): string => `${_wsBase()}${path}`;

/** Upload a file with FormData */
export const uploadFile = (
  endpoint: string,
  file: File,
  extraFields?: Record<string, string>,
) => {
  const formData = new FormData();
  formData.append('file', file);
  if (extraFields) {
    Object.entries(extraFields).forEach(([k, v]) => formData.append(k, v));
  }
  return api.post(endpoint, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

/** Upload multiple files */
export const uploadFiles = (
  endpoint: string,
  files: File[],
  extraFields?: Record<string, string>,
) => {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  if (extraFields) {
    Object.entries(extraFields).forEach(([k, v]) => formData.append(k, v));
  }
  return api.post(endpoint, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 min for large batches
  });
};
