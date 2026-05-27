/**
 * useGpuStatus — WebSocket-driven GPU stats with system store integration.
 */

'use client';

import { useCallback, useMemo } from 'react';
import { useWebSocket } from './useWebSocket';
import { useSystemStore } from '@/store/systemStore';
import type { WsGpuStats, GpuStats } from '@/lib/types';

let clientIdCounter = 0;
const GPU_CLIENT_ID = `gpu-monitor-${++clientIdCounter}`;

export function useGpuStatus() {
  const { gpu, cpuRam, wsConnected, setGpuStats, setCpuRam, setWsConnected } =
    useSystemStore();

  const handleMessage = useCallback(
    (data: unknown) => {
      const msg = data as WsGpuStats;
      if (msg.type === 'gpu_stats' && msg.gpu) {
        setGpuStats(msg.gpu);
      }
      // Also handle cpu_ram if present
      if ('cpu_ram' in msg && msg.cpu_ram) {
        setCpuRam(msg.cpu_ram as never);
      }
    },
    [setGpuStats, setCpuRam],
  );

  const { connected } = useWebSocket('/ws/system', GPU_CLIENT_ID, {
    onMessage: handleMessage,
    onConnect: () => setWsConnected(true),
    onDisconnect: () => setWsConnected(false),
  });

  const vramUsedPct = gpu?.vram_used_pct ?? 0;
  const gpuUtilPct = gpu?.gpu_utilization_pct ?? null;

  return {
    gpu,
    cpuRam,
    wsConnected: connected,
    vramUsedPct,
    gpuUtilPct,
    deviceName: gpu?.device_name ?? 'Unknown GPU',
    vramTotal: gpu?.vram_total_gb ?? 0,
    vramUsed: gpu?.vram_used_gb ?? 0,
    vramFree: gpu?.vram_free_gb ?? 0,
    temperature: gpu?.temperature_c ?? null,
    powerDraw: gpu?.power_draw_w ?? null,
  };
}
