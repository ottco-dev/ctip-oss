'use client';

import { useEffect, useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Database, ChevronDown, ChevronUp, CheckCircle2, AlertCircle, Loader2, Tag, Activity } from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { useTrainingStore } from '@/store/trainingStore';
import type { LSDataset, PrepareDatasetRequest, PrepareDatasetResponse, PrepareDatasetStartedResponse } from '@/lib/types';

interface LabelStudioDatasetPickerProps {
  onDatasetReady: (yamlPath: string, info: PrepareDatasetResponse) => void;
}

const CLASS_COLORS: Record<string, string> = {
  stalked: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  sessile: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  bulbous: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  'non-glandular': 'bg-red-500/20 text-red-300 border-red-500/30',
};

export function LabelStudioDatasetPicker({ onDatasetReady }: LabelStudioDatasetPickerProps) {
  const [expanded, setExpanded] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [usePredictions, setUsePredictions] = useState(false);
  const [trainRatio, setTrainRatio] = useState(0.70);
  const [valRatio, setValRatio] = useState(0.15);
  const [lastResult, setLastResult] = useState<PrepareDatasetResponse | null>(null);
  const [currentPrepareId, setCurrentPrepareId] = useState<string | null>(null);
  const [wsErrorMsg, setWsErrorMsg] = useState<string | null>(null);

  const { datasetReadyMap } = useTrainingStore();

  // Watch for dataset_ready WS event
  useEffect(() => {
    if (!currentPrepareId) return;
    const result = datasetReadyMap[currentPrepareId];
    if (!result) return;

    setCurrentPrepareId(null);

    if (result.success && result.dataset_yaml) {
      setWsErrorMsg(null);
      const resp: PrepareDatasetResponse = {
        dataset_yaml: result.dataset_yaml!,
        dataset_dir: result.dataset_dir ?? '',
        total_tasks: result.total_tasks ?? 0,
        exported_tasks: result.exported_tasks ?? 0,
        skipped_tasks: result.skipped_tasks ?? 0,
        train_count: result.train_count ?? 0,
        val_count: result.val_count ?? 0,
        test_count: result.test_count ?? 0,
        classes: result.classes ?? [],
        warnings: result.warnings ?? [],
      };
      setLastResult(resp);
      onDatasetReady(resp.dataset_yaml, resp);
    } else {
      setWsErrorMsg(result.error ?? 'Export failed — see Training Log for details');
    }
  }, [currentPrepareId, datasetReadyMap, onDatasetReady]);

  const { data: projects, isLoading, error } = useQuery<LSDataset[]>({
    queryKey: ['ls-datasets'],
    queryFn: () => api.get('/training/ls-datasets').then((r) => r.data),
    enabled: expanded,
    staleTime: 30_000,
  });

  const prepareMutation = useMutation({
    mutationFn: (req: PrepareDatasetRequest) =>
      api.post('/training/prepare-ls-dataset', req).then((r) => r.data as PrepareDatasetStartedResponse),
    onSuccess: (data) => {
      setCurrentPrepareId(data.prepare_id);
    },
  });

  const selected = projects?.find((p) => p.project_id === selectedId);
  const isPreparing = prepareMutation.isPending || (currentPrepareId !== null && !datasetReadyMap[currentPrepareId]);
  const prepareError = wsErrorMsg
    ?? (prepareMutation.isError
      ? String((prepareMutation.error as any)?.response?.data?.detail ?? prepareMutation.error)
      : null);

  const handlePrepare = () => {
    if (!selectedId) return;
    setLastResult(null);
    setWsErrorMsg(null);
    prepareMutation.reset();
    prepareMutation.mutate({
      project_id: selectedId,
      use_predictions: usePredictions,
      train_ratio: trainRatio,
      val_ratio: valRatio,
      seed: 42,
    });
  };

  const testPct = Math.max(0, 1 - trainRatio - valRatio);

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <button
        className="w-full flex items-center justify-between px-4 py-2.5 bg-surface-secondary hover:bg-surface-tertiary transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-text-secondary" />
          <span className="text-sm font-medium text-text-primary">Label Studio Dataset</span>
          {lastResult && (
            <span className="text-xs text-status-success font-mono ml-1">
              ✓ {lastResult.exported_tasks} tasks ready
            </span>
          )}
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-text-muted" />
        ) : (
          <ChevronDown className="w-4 h-4 text-text-muted" />
        )}
      </button>

      {expanded && (
        <div className="p-4 space-y-4 border-t border-border">

          {/* Project list */}
          {isLoading && (
            <div className="flex items-center gap-2 text-sm text-text-muted">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading Label Studio projects…
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 text-xs text-status-error bg-status-error/10 rounded p-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              Cannot reach Label Studio. Is it running on port 3005?
            </div>
          )}

          {projects && projects.length === 0 && (
            <p className="text-xs text-text-muted">No projects with tasks found in Label Studio.</p>
          )}

          {projects && projects.length > 0 && (
            <div className="space-y-2">
              {projects.map((p) => {
                const isSelected = p.project_id === selectedId;
                const hasAnnotations = p.annotation_count > 0;
                const hasPredictions = p.prediction_count > 0;

                return (
                  <button
                    key={p.project_id}
                    onClick={() => setSelectedId(isSelected ? null : p.project_id)}
                    className={cn(
                      'w-full text-left rounded-lg border p-3 transition-colors',
                      isSelected
                        ? 'border-accent bg-accent/10'
                        : 'border-border bg-surface-secondary hover:border-border-hover hover:bg-surface-tertiary',
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-accent flex-shrink-0" />}
                          <span className="text-sm font-medium text-text-primary truncate">{p.title}</span>
                        </div>
                        {p.description && (
                          <p className="text-xs text-text-muted mt-0.5 line-clamp-1">{p.description}</p>
                        )}
                      </div>
                      <div className="flex flex-col items-end gap-1 text-xs flex-shrink-0">
                        <span className="text-text-secondary font-mono">{p.task_count} tasks</span>
                        <div className="flex gap-1.5">
                          {hasAnnotations && (
                            <span className="badge badge-success">
                              {p.annotation_count} annotated
                            </span>
                          )}
                          {hasPredictions && (
                            <span className="badge badge-info">
                              {p.prediction_count} pre-ann
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {/* Options (shown when project selected) */}
          {selected && (
            <div className="space-y-3 pt-2 border-t border-border">

              {/* Annotation source toggle */}
              <div className="space-y-1.5">
                <label className="text-xs text-text-secondary font-medium">Annotation Source</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setUsePredictions(false)}
                    className={cn(
                      'text-xs px-3 py-2 rounded-lg border transition-colors text-left',
                      !usePredictions
                        ? 'border-accent bg-accent/10 text-accent'
                        : 'border-border text-text-secondary hover:border-border-hover',
                    )}
                  >
                    <div className="font-medium">Human annotations</div>
                    <div className="text-text-muted mt-0.5">
                      {selected.annotation_count} tasks confirmed
                    </div>
                  </button>
                  <button
                    onClick={() => setUsePredictions(true)}
                    className={cn(
                      'text-xs px-3 py-2 rounded-lg border transition-colors text-left',
                      usePredictions
                        ? 'border-accent bg-accent/10 text-accent'
                        : 'border-border text-text-secondary hover:border-border-hover',
                    )}
                  >
                    <div className="font-medium">Pre-annotations</div>
                    <div className="text-text-muted mt-0.5">
                      {selected.prediction_count} YOLO suggestions
                    </div>
                  </button>
                </div>
                {!usePredictions && selected.annotation_count === 0 && (
                  <p className="text-xs text-status-warning">
                    No human annotations yet. Use pre-annotations or annotate tasks in Label Studio first.
                  </p>
                )}
              </div>

              {/* Split ratios */}
              <div className="space-y-1.5">
                <label className="text-xs text-text-secondary font-medium">Train / Val / Test Split</label>
                <div className="flex gap-2 items-center">
                  <div className="flex-1">
                    <div className="flex justify-between text-[10px] text-text-muted mb-1">
                      <span>Train</span>
                      <span className="font-mono text-accent">{(trainRatio * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      className="w-full accent-accent h-1.5"
                      min={0.3}
                      max={0.85}
                      step={0.05}
                      value={trainRatio}
                      onChange={(e) => {
                        const t = parseFloat(e.target.value);
                        setTrainRatio(t);
                        if (t + valRatio > 0.95) setValRatio(0.95 - t);
                      }}
                    />
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between text-[10px] text-text-muted mb-1">
                      <span>Val</span>
                      <span className="font-mono text-accent">{(valRatio * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      className="w-full accent-accent h-1.5"
                      min={0.05}
                      max={0.35}
                      step={0.05}
                      value={valRatio}
                      onChange={(e) => setValRatio(parseFloat(e.target.value))}
                    />
                  </div>
                </div>

                {/* Visual split bar */}
                <div className="flex rounded overflow-hidden h-1.5 gap-px">
                  <div className="bg-accent" style={{ width: `${trainRatio * 100}%` }} />
                  <div className="bg-blue-400" style={{ width: `${valRatio * 100}%` }} />
                  <div className="bg-surface-tertiary flex-1" />
                </div>
                <div className="flex justify-between text-[10px] text-text-muted">
                  <span>
                    Train {Math.round((usePredictions ? selected.prediction_count : selected.annotation_count || selected.task_count) * trainRatio)} tasks
                  </span>
                  <span>
                    Val {Math.round((usePredictions ? selected.prediction_count : selected.annotation_count || selected.task_count) * valRatio)} tasks
                  </span>
                  <span>
                    Test {Math.round((usePredictions ? selected.prediction_count : selected.annotation_count || selected.task_count) * testPct)} tasks
                  </span>
                </div>
              </div>

              {/* Prepare button */}
              <button
                className="btn-primary w-full flex items-center justify-center gap-2 text-sm"
                onClick={handlePrepare}
                disabled={isPreparing || (!usePredictions && selected.annotation_count === 0)}
              >
                {isPreparing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Preparing dataset…
                  </>
                ) : (
                  <>
                    <Database className="w-4 h-4" />
                    Prepare Dataset
                  </>
                )}
              </button>

              {/* Live progress hint */}
              {isPreparing && (
                <div className="flex items-center gap-1.5 text-xs text-text-muted animate-pulse">
                  <Activity className="w-3 h-3" />
                  Progress streaming to Training Log below…
                </div>
              )}

              {/* Error */}
              {prepareError && (
                <div className="text-xs text-status-error bg-status-error/10 rounded p-2">
                  {prepareError}
                </div>
              )}

              {/* Success */}
              {lastResult && !prepareMutation.isPending && (
                <div className="bg-status-success/10 border border-status-success/20 rounded-lg p-3 space-y-2">
                  <div className="flex items-center gap-1.5 text-xs text-status-success font-medium">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Dataset ready — {lastResult.exported_tasks} tasks exported
                  </div>
                  <div className="font-mono text-[10px] text-text-muted break-all">
                    {lastResult.dataset_yaml}
                  </div>
                  <div className="flex gap-1 flex-wrap">
                    {lastResult.classes.map((cls) => (
                      <span
                        key={cls}
                        className={cn(
                          'inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border',
                          CLASS_COLORS[cls] ?? 'bg-surface-tertiary text-text-secondary border-border',
                        )}
                      >
                        <Tag className="w-2.5 h-2.5" />
                        {cls}
                      </span>
                    ))}
                  </div>
                  {lastResult.skipped_tasks > 0 && (
                    <div className="text-[10px] text-status-warning">
                      {lastResult.skipped_tasks} tasks skipped (no valid bounding boxes or unresolvable images)
                    </div>
                  )}
                  {lastResult.warnings.length > 0 && (
                    <details className="text-[10px] text-text-muted">
                      <summary className="cursor-pointer text-status-warning">
                        {lastResult.warnings.length} warnings
                      </summary>
                      <ul className="mt-1 space-y-0.5 list-disc list-inside">
                        {lastResult.warnings.slice(0, 10).map((w, i) => (
                          <li key={i}>{w}</li>
                        ))}
                        {lastResult.warnings.length > 10 && (
                          <li>…and {lastResult.warnings.length - 10} more</li>
                        )}
                      </ul>
                    </details>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
