/**
 * trainingStore — Zustand store for active training run state.
 */

import { create } from 'zustand';
import type { MetricPoint, LogLevel, WsDatasetReady } from '@/lib/types';

export interface LogEntry {
  ts: number;
  line: string;
  level: LogLevel;
  run_id: string;
}

const MAX_LOG_LINES = 2000;

interface TrainingState {
  activeRunUuid: string | null;
  activeRunStatus: string | null;
  currentEpoch: number;
  totalEpochs: number;
  liveMetrics: MetricPoint[];
  bestMap50: number;
  bestMap50Epoch: number;
  logLines: LogEntry[];
  datasetReadyMap: Record<string, WsDatasetReady>;

  setActiveRun: (runUuid: string, totalEpochs: number) => void;
  clearActiveRun: () => void;
  addMetrics: (epoch: number, metrics: Record<string, number>) => void;
  addLogLine: (entry: LogEntry) => void;
  clearLog: () => void;
  setStatus: (status: string) => void;
  setDatasetReady: (result: WsDatasetReady) => void;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  activeRunUuid: null,
  activeRunStatus: null,
  currentEpoch: 0,
  totalEpochs: 0,
  liveMetrics: [],
  bestMap50: 0,
  bestMap50Epoch: 0,
  logLines: [],
  datasetReadyMap: {},

  setActiveRun: (runUuid, totalEpochs) =>
    set({
      activeRunUuid: runUuid,
      currentEpoch: 0,
      totalEpochs,
      liveMetrics: [],
      bestMap50: 0,
      activeRunStatus: 'running',
      logLines: [],
    }),

  clearActiveRun: () =>
    set({
      activeRunUuid: null,
      activeRunStatus: null,
      currentEpoch: 0,
    }),

  addMetrics: (epoch, metrics) => {
    const newPoints: MetricPoint[] = Object.entries(metrics).map(([key, value]) => ({
      epoch,
      key,
      value,
    }));

    const { bestMap50 } = get();
    const currentMap50 = metrics['metrics/mAP50(B)'] ?? metrics['val_map50'] ?? 0;
    const newBest = currentMap50 > bestMap50;

    set((state) => ({
      currentEpoch: epoch,
      liveMetrics: [...state.liveMetrics, ...newPoints],
      bestMap50: newBest ? currentMap50 : state.bestMap50,
      bestMap50Epoch: newBest ? epoch : state.bestMap50Epoch,
    }));
  },

  addLogLine: (entry) =>
    set((state) => ({
      logLines:
        state.logLines.length >= MAX_LOG_LINES
          ? [...state.logLines.slice(-MAX_LOG_LINES + 1), entry]
          : [...state.logLines, entry],
    })),

  clearLog: () => set({ logLines: [] }),

  setStatus: (status) => set({ activeRunStatus: status }),

  setDatasetReady: (result) =>
    set((state) => ({
      datasetReadyMap: { ...state.datasetReadyMap, [result.prepare_id]: result },
    })),
}));
