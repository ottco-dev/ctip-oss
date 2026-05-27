/**
 * API client — axios instance with base URL and error handling.
 */

import axios, { AxiosError, type AxiosInstance } from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

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

export const wsUrl = (path: string): string => `${WS_BASE}${path}`;

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
