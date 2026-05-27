"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { DatasetSummary, DatasetStats, SampleSummary } from "@/lib/types";

/**
 * Fetch all datasets.
 */
export function useDatasets() {
  return useQuery<DatasetSummary[]>({
    queryKey: ["datasets"],
    queryFn: async () => {
      const r = await api.get("/datasets");
      // Handle both [...] and {datasets: [...]} response shapes
      return Array.isArray(r.data) ? r.data : r.data?.datasets ?? r.data?.items ?? [];
    },
    staleTime: 30_000,
    refetchInterval: 30_000,
  });
}

/**
 * Fetch a single dataset by ID.
 */
export function useDataset(id: number | null) {
  return useQuery<DatasetSummary>({
    queryKey: ["datasets", id],
    queryFn: () => api.get(`/datasets/${id}`).then((r) => r.data),
    enabled: id !== null,
    staleTime: 30_000,
  });
}

/**
 * Fetch stats for a dataset.
 */
export function useDatasetStats(id: number | null) {
  return useQuery<DatasetStats>({
    queryKey: ["datasets", id, "stats"],
    queryFn: () => api.get(`/datasets/${id}/stats`).then((r) => r.data),
    enabled: id !== null,
    staleTime: 60_000,
  });
}

/**
 * Fetch samples for a dataset.
 */
export function useDatasetSamples(id: number | null, page = 1, pageSize = 50) {
  return useQuery<SampleSummary[]>({
    queryKey: ["datasets", id, "samples", page, pageSize],
    queryFn: async () => {
      const r = await api.get(`/datasets/${id}/samples`, {
        params: { page, page_size: pageSize },
      });
      return Array.isArray(r.data) ? r.data : r.data?.samples ?? r.data?.items ?? [];
    },
    enabled: id !== null,
    staleTime: 60_000,
  });
}
