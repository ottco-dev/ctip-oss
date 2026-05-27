"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

interface QueueItem {
  id: number;
  image_path?: string;
  filename?: string;
  maturity_stage?: string;
  clear_fraction?: number;
  cloudy_fraction?: number;
  amber_fraction?: number;
  vlm_confidence?: number;
  confidence?: number;
  hallucination_flags?: string[];
  review_priority?: number;
  priority?: number;
  queued_at?: string;
  created_at?: string;
  vlm_backend?: string;
  status?: string;
}

interface QueueResponse {
  items?: QueueItem[];
  queue?: QueueItem[];
  stats?: {
    total_pending?: number;
    total_reviewed?: number;
    throughput_per_hour?: number;
    high_priority_count?: number;
  };
}

interface AutoLabelParams {
  dataset_id: number;
  backend?: string;
  batch_size?: number;
}

/**
 * Fetch the annotation review queue.
 */
export function useQueue() {
  return useQuery<QueueResponse>({
    queryKey: ["annotation-queue"],
    queryFn: () => api.get("/annotation/queue").then((r) => r.data),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

/**
 * Approve a queue item by ID (PUT /annotation/queue/{id}).
 */
export function useApproveItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (itemId: number) =>
      api.put(`/annotation/queue/${itemId}`, { status: "approved" }).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["annotation-queue"] });
    },
  });
}

/**
 * Reject a queue item by ID (PUT /annotation/queue/{id}).
 */
export function useRejectItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (itemId: number) =>
      api.put(`/annotation/queue/${itemId}`, { status: "rejected" }).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["annotation-queue"] });
    },
  });
}

/**
 * Trigger VLM auto-labeling for a dataset (POST /annotation/auto-label).
 */
export function useAutoLabel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (params: AutoLabelParams) =>
      api.post("/annotation/auto-label", params).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["annotation-queue"] });
      queryClient.invalidateQueries({ queryKey: ["annotation-jobs"] });
    },
  });
}
