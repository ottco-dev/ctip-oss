/**
 * TypeScript types — mirrors Python Pydantic schemas.
 */

// ── SYSTEM ──────────────────────────────────────────────────────

export interface GpuStats {
  available: boolean;
  device_name?: string;
  vram_total_gb?: number;
  vram_used_gb?: number;
  vram_reserved_gb?: number;
  vram_free_gb?: number;
  vram_used_pct?: number;
  gpu_utilization_pct?: number | null;
  temperature_c?: number;
  power_draw_w?: number;
  reason?: string;
}

export interface CpuRamStats {
  cpu_count: number;
  cpu_utilization_pct: number;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_free_gb: number;
  ram_used_pct: number;
  disk_total_gb: number;
  disk_free_gb: number;
}

export interface SystemInfo {
  timestamp: number;
  gpu: GpuStats;
  cpu_ram: CpuRamStats;
  config: Record<string, unknown>;
}

// ── TRAINING ────────────────────────────────────────────────────

export type RunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'stopped';

export interface RunSummary {
  id: number;
  run_uuid: string;
  experiment_id: number;
  model_variant: string;
  status: RunStatus;
  best_map50: number;
  best_map50_95: number;
  best_precision: number;
  best_recall: number;
  best_epoch: number;
  total_epochs: number;
  started_at: number | null;
  finished_at: number | null;
  duration_s: number | null;
}

export interface MetricPoint {
  epoch: number;
  key: string;
  value: number;
}

export interface TrainingStartRequest {
  // Core
  experiment_name: string;
  model_variant: string;
  data_yaml: string;
  epochs: number;
  batch_size: number;
  imgsz: number;
  amp: boolean;
  seed: number;
  notes?: string;

  // LR schedule
  lr0: number;
  lrf: number;
  warmup_epochs: number;
  cos_lr: boolean;

  // Regularisation
  weight_decay: number;
  momentum: number;

  // Early stopping
  patience: number;

  // Augmentation
  augment: boolean;
  mosaic: number;
  close_mosaic: number;
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  degrees: number;
  scale: number;
  flipud: number;
  fliplr: number;
}

// ── DATASETS ────────────────────────────────────────────────────

export interface DatasetSummary {
  id: number;
  name: string;
  description: string;
  num_samples: number;
  num_annotated: number;
  num_reviewed: number;
  class_names: string[];
  version: string;
  status: string;
  created_at: number;
}

export interface SampleSummary {
  id: number;
  dataset_id?: number;
  filename: string;
  width: number;
  height: number;
  quality_score: number;
  focus_score: number;
  split: string;
  num_annotations: number;
  annotation_source: string;
  reviewed?: boolean;
  created_at: number;
}

export interface DatasetStats {
  dataset_id: number;
  num_samples: number;
  num_annotated: number;
  num_reviewed: number;
  split_distribution: Record<string, number>;
  quality: {
    mean: number;
    std: number;
    min: number;
    max: number;
    usable: number;
  };
  class_distribution: Record<string, number>;
  total_annotations: number;
}

// ── DETECTION ───────────────────────────────────────────────────

export interface DetectionBox {
  id: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  calibrated_confidence: number | null;
  uncertainty: number | null;
  trichome_type: string;
  is_uncertain: boolean;
}

export interface DetectionResponse {
  image_id: string;
  detections: DetectionBox[];
  num_detections: number;
  mean_confidence: number;
  processing_time_ms: number;
  model_id: string;
  trichome_counts: Record<string, number>;
}

// ── VLM LABELING ─────────────────────────────────────────────────

export type MaturityStage =
  | 'clear'
  | 'cloudy'
  | 'amber'
  | 'cloudy_amber_mix'
  | 'degraded'
  | 'unknown';

export interface MaturityLabelResponse {
  label_id: string;
  maturity_stage: MaturityStage | null;
  confidence: number;
  amber_fraction: number | null;
  cloudy_fraction: number | null;
  clear_fraction: number | null;
  observations: string | null;
  image_quality: string | null;
  hallucination_flags: string[];
  is_flagged: boolean;
  review_priority: number;
  inference_time_s: number;
  vlm_model: string;
  annotation_source: string;
  scientific_caveat: string;
}

// ── JOBS ────────────────────────────────────────────────────────

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface JobSummary {
  job_uuid: string;
  job_type: string;
  status: JobStatus;
  progress: number;
  progress_pct: number;
  total_items: number | null;
  processed_items: number;
  error_message: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  duration_s: number | null;
}

// ── WEBSOCKET EVENTS ─────────────────────────────────────────────

export interface WsTrainingMetrics {
  type: 'training_metrics';
  run_id: string;
  epoch: number;
  metrics: Record<string, number>;
  _ts: number;
}

export interface WsJobUpdate {
  type: 'job_update';
  job_uuid: string;
  status: JobStatus;
  progress: number;
  progress_pct: number;
  message: string;
  _ts: number;
}

export interface WsGpuStats {
  type: 'gpu_stats';
  gpu: GpuStats;
  timestamp: number;
}

export type LogLevel = 'info' | 'success' | 'warning' | 'error' | 'dim' | 'header';

export interface WsTrainingLog {
  type: 'training_log';
  run_id: string;
  line: string;
  level: LogLevel;
  _ts: number;
}

export interface WsDatasetReady {
  type: 'dataset_ready';
  prepare_id: string;
  success: boolean;
  error?: string;
  dataset_yaml?: string;
  dataset_dir?: string;
  total_tasks?: number;
  exported_tasks?: number;
  skipped_tasks?: number;
  train_count?: number;
  val_count?: number;
  test_count?: number;
  classes?: string[];
  warnings?: string[];
  _ts: number;
}

export type WsEvent = WsTrainingMetrics | WsJobUpdate | WsGpuStats | WsTrainingLog | WsDatasetReady;

// ── LABEL STUDIO DATASET INTEGRATION ────────────────────────────

export interface LSDataset {
  project_id: number;
  title: string;
  task_count: number;
  annotation_count: number;
  prediction_count: number;
  description: string;
}

export interface PrepareDatasetRequest {
  project_id: number;
  use_predictions: boolean;
  train_ratio: number;
  val_ratio: number;
  seed: number;
}

export interface PrepareDatasetResponse {
  dataset_yaml: string;
  dataset_dir: string;
  total_tasks: number;
  exported_tasks: number;
  skipped_tasks: number;
  train_count: number;
  val_count: number;
  test_count: number;
  classes: string[];
  warnings: string[];
}

export interface PrepareDatasetStartedResponse {
  prepare_id: string;
  status: string;
  message: string;
}
