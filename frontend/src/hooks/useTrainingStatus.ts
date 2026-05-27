/**
 * useTrainingStatus — Live training metrics via WebSocket.
 */

'use client';

import { useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { useTrainingStore } from '@/store/trainingStore';
import type { WsTrainingMetrics } from '@/lib/types';

let trainingClientCounter = 0;
const TRAINING_CLIENT_ID = `training-${++trainingClientCounter}`;

export function useTrainingStatus() {
  const { activeRunUuid, currentEpoch, totalEpochs, liveMetrics, bestMap50, addMetrics, setStatus } =
    useTrainingStore();

  const handleMessage = useCallback(
    (data: unknown) => {
      const msg = data as WsTrainingMetrics;
      if (msg.type === 'training_metrics') {
        addMetrics(msg.epoch, msg.metrics);
      }
    },
    [addMetrics],
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
