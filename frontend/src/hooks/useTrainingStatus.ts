/**
 * useTrainingStatus — Live training metrics + log lines via WebSocket.
 */

'use client';

import { useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { useTrainingStore } from '@/store/trainingStore';
import type { WsTrainingMetrics, WsTrainingLog, WsDatasetReady } from '@/lib/types';

const TRAINING_CLIENT_ID = 'training-log';

export function useTrainingStatus() {
  const {
    activeRunUuid, currentEpoch, totalEpochs, liveMetrics, bestMap50,
    addMetrics, setStatus, addLogLine, setDatasetReady,
  } = useTrainingStore();

  const handleMessage = useCallback(
    (data: unknown) => {
      const msg = data as { type: string };
      if (msg.type === 'training_metrics') {
        const m = msg as WsTrainingMetrics;
        addMetrics(m.epoch, m.metrics);
      } else if (msg.type === 'training_log') {
        const log = msg as WsTrainingLog;
        addLogLine({ ts: log._ts ?? Date.now() / 1000, line: log.line, level: log.level, run_id: log.run_id });
      } else if (msg.type === 'dataset_ready') {
        setDatasetReady(msg as WsDatasetReady);
      }
    },
    [addMetrics, addLogLine, setDatasetReady],
  );

  const { connected } = useWebSocket('/ws/training', TRAINING_CLIENT_ID, {
    onMessage: handleMessage,
    enabled: true,
  });

  // Derive chart-friendly data
  const trainLossData = liveMetrics
    .filter((m) => m.key === 'train_loss' || m.key === 'train/box_loss')
    .map((m) => ({ epoch: m.epoch, value: m.value }));

  const valMap50Data = liveMetrics
    .filter((m) => m.key.includes('mAP50') || m.key === 'val_map50')
    .map((m) => ({ epoch: m.epoch, value: m.value }));

  const progressPct = totalEpochs > 0 ? (currentEpoch / totalEpochs) * 100 : 0;

  return {
    activeRunUuid,
    currentEpoch,
    totalEpochs,
    progressPct,
    bestMap50,
    liveMetrics,
    trainLossData,
    valMap50Data,
    wsConnected: connected,
    isTraining: activeRunUuid !== null,
  };
}
