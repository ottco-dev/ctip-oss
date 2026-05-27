/**
 * systemStore — Zustand store for GPU stats and queue status.
 */

import { create } from 'zustand';
import type { GpuStats, CpuRamStats } from '@/lib/types';

interface SystemState {
  gpu: GpuStats | null;
  cpuRam: CpuRamStats | null;
  wsConnected: boolean;
  gpuTaskRunning: string | null;
  queueDepth: number;
  lastUpdated: number;

  setGpuStats: (gpu: GpuStats) => void;
  setCpuRam: (cpuRam: CpuRamStats) => void;
  setWsConnected: (connected: boolean) => void;
  setGpuTask: (jobId: string | null) => void;
  setQueueDepth: (depth: number) => void;
}

export const useSystemStore = create<SystemState>((set) => ({
  gpu: null,
  cpuRam: null,
  wsConnected: false,
  gpuTaskRunning: null,
  queueDepth: 0,
  lastUpdated: 0,

  setGpuStats: (gpu) => set({ gpu, lastUpdated: Date.now() }),
  setCpuRam: (cpuRam) => set({ cpuRam }),
  setWsConnected: (wsConnected) => set({ wsConnected }),
  setGpuTask: (gpuTaskRunning) => set({ gpuTaskRunning }),
  setQueueDepth: (queueDepth) => set({ queueDepth }),
}));
