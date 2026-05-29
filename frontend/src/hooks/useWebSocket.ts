'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { wsUrl } from '@/lib/api';

interface UseWebSocketOptions {
  onMessage?: (data: unknown) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  enabled?: boolean;
  maxReconnectAttempts?: number;
  reconnectBaseDelayMs?: number;
  initialDelayMs?: number; // Avoids React StrictMode double-mount false error
}

interface UseWebSocketResult {
  connected: boolean;
  reconnectAttempts: number;
  send: (data: unknown) => void;
}

export function useWebSocket(
  path: string,
  clientId: string,
  options: UseWebSocketOptions = {},
): UseWebSocketResult {
  const {
    enabled = true,
    maxReconnectAttempts = 15,
    reconnectBaseDelayMs = 1500,
    initialDelayMs = 200, // Short delay avoids StrictMode double-connect error
  } = options;

  // Keep callbacks in refs so changing them never triggers reconnects
  const onMessageRef = useRef(options.onMessage);
  const onConnectRef = useRef(options.onConnect);
  const onDisconnectRef = useRef(options.onDisconnect);
  useEffect(() => {
    onMessageRef.current = options.onMessage;
    onConnectRef.current = options.onConnect;
    onDisconnectRef.current = options.onDisconnect;
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const initialTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const [connected, setConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const connect = useCallback(() => {
    if (!mountedRef.current || !enabled) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    const url = `${wsUrl(path)}?client_id=${encodeURIComponent(clientId)}`;

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      return; // WebSocket constructor failed (SSR or invalid URL)
    }
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return; }
      reconnectAttemptRef.current = 0;
      setConnected(true);
      setReconnectAttempts(0);
      onConnectRef.current?.();
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current) return;
      try {
        onMessageRef.current?.(JSON.parse(event.data));
      } catch {
        // Non-JSON — ignore
      }
    };

    ws.onclose = (ev) => {
      if (!mountedRef.current) return;
      // 1000 = normal close (we closed it intentionally)
      if (ev.code === 1000) return;
      setConnected(false);
      onDisconnectRef.current?.();
      wsRef.current = null;

      if (reconnectAttemptRef.current < maxReconnectAttempts && enabled) {
        const delay = Math.min(
          reconnectBaseDelayMs * Math.pow(1.8, reconnectAttemptRef.current),
          30_000,
        );
        reconnectAttemptRef.current++;
        setReconnectAttempts(reconnectAttemptRef.current);
        reconnectTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current && enabled) connect();
        }, delay);
      }
    };

    ws.onerror = () => {
      // Let onclose handle reconnect
      ws.close();
    };
  }, [path, clientId, enabled, maxReconnectAttempts, reconnectBaseDelayMs]);

  useEffect(() => {
    mountedRef.current = true;
    if (enabled) {
      // Small initial delay avoids React StrictMode double-mount noise
      initialTimeoutRef.current = setTimeout(connect, initialDelayMs);
    }
    return () => {
      mountedRef.current = false;
      if (initialTimeoutRef.current) clearTimeout(initialTimeoutRef.current);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close(1000, 'unmount');
        wsRef.current = null;
      }
      setConnected(false);
    };
  }, [connect, enabled, initialDelayMs]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, reconnectAttempts, send };
}
