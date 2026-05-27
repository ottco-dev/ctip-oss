"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

interface ModelVersion {
  id: number;
  name: string;
  model_type: string;
  framework: string;
  variant: string;
  file_path: string | null;
  file_size_bytes: number | null;
  vram_required_gb: number | null;
  metrics: Record<string, number>;
  created_at: string;
  is_downloaded: boolean;
  is_active: boolean;
  description?: string;
  source_url?: string;
}

/**
 * Fetch all registered models.
 */
export function useModels() {
  return useQuery<ModelVersion[]>({
    queryKey: ["models"],
    queryFn: async () => {
      const r = await api.get("/models");
      return Array.isArray(r.data) ? r.data : r.data?.models ?? [];
    },
    staleTime: 60_000,
  });
}

/**
 * Fetch a single model by ID.
 */
export function useModel(id: number | null) {
  return useQuery<ModelVersion>({
    queryKey: ["models", id],
    queryFn: () => api.get(`/models/${id}`).then((r) => r.data),
    enabled: id !== null,
    staleTime: 60_000,
  });
}

/**
 * Mutation to activate a model (PUT /models/{id}/activate).
 * Invalidates the models query on success.
 */
export function useActivateModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => api.put(`/models/${id}/activate`).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

/**
 * Mutation to trigger model download (POST /models/{id}/download).
 */
export function useDownloadModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => api.post(`/models/${id}/download`).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}
