"use client";

import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, CheckCircle2, AlertCircle, Loader2, ImageIcon } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";
import { uploadFiles } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface UploadFile {
  id: string;
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  progress: number;
  errorMessage?: string;
}

interface UploadZoneProps {
  datasetId: number;
  onUploadComplete: (count: number) => void;
  accept?: Record<string, string[]>;
  maxFileSizeMb?: number;
}

// ---------------------------------------------------------------------------
// Single file row
// ---------------------------------------------------------------------------

function FileRow({ item, onRemove }: { item: UploadFile; onRemove: () => void }) {
  const ext = item.file.name.split(".").pop()?.toLowerCase() ?? "";

  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)]">
      {/* Icon */}
      <div className="flex-shrink-0">
        {item.status === "done" ? (
          <CheckCircle2 className="w-4 h-4 text-green-400" />
        ) : item.status === "error" ? (
          <AlertCircle className="w-4 h-4 text-red-400" />
        ) : item.status === "uploading" ? (
          <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
        ) : (
          <ImageIcon className="w-4 h-4 text-[var(--color-text-muted)]" />
        )}
      </div>

      {/* Name + size */}
      <div className="flex-1 min-w-0">
        <p className="text-sm text-[var(--color-text-secondary)] truncate">{item.file.name}</p>
        <p className="text-xs text-[var(--color-text-muted)]">{formatBytes(item.file.size)}</p>
      </div>

      {/* Progress bar (uploading) */}
      {item.status === "uploading" && (
        <div className="w-24 h-1.5 bg-[var(--color-border)] rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${item.progress}%` }}
          />
        </div>
      )}

      {/* Error message */}
      {item.status === "error" && item.errorMessage && (
        <span className="text-xs text-red-400 max-w-[100px] truncate" title={item.errorMessage}>
          {item.errorMessage}
        </span>
      )}

      {/* Remove (only when pending or error) */}
      {(item.status === "pending" || item.status === "error") && (
        <button
          onClick={onRemove}
          className="flex-shrink-0 text-[var(--color-text-muted)] hover:text-red-400 transition-colors"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// UploadZone
// ---------------------------------------------------------------------------

export function UploadZone({
  datasetId,
  onUploadComplete,
  accept = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/tiff": [".tif", ".tiff"],
    "image/bmp": [".bmp"],
  },
  maxFileSizeMb = 100,
}: UploadZoneProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStats, setUploadStats] = useState<{
    total: number;
    done: number;
    errors: number;
  } | null>(null);

  // ---------------------------------------------------------------------------
  // Dropzone
  // ---------------------------------------------------------------------------

  const onDrop = useCallback(
    (accepted: File[]) => {
      const newItems: UploadFile[] = accepted.map((f) => ({
        id: `${f.name}-${f.size}-${Date.now()}-${Math.random()}`,
        file: f,
        status: "pending",
        progress: 0,
      }));
      setFiles((prev) => [...prev, ...newItems]);
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept,
    maxSize: maxFileSizeMb * 1024 * 1024,
    multiple: true,
  });

  const removeFile = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const clearDone = useCallback(() => {
    setFiles((prev) => prev.filter((f) => f.status !== "done"));
    setUploadStats(null);
  }, []);

  // ---------------------------------------------------------------------------
  // Upload handler
  // ---------------------------------------------------------------------------

  const handleUpload = useCallback(async () => {
    const pending = files.filter((f) => f.status === "pending");
    if (pending.length === 0 || isUploading) return;

    setIsUploading(true);
    setUploadStats({ total: pending.length, done: 0, errors: 0 });

    let doneCount = 0;
    let errorCount = 0;

    // Upload in batches of 10 to avoid overwhelming the server
    const BATCH_SIZE = 10;
    for (let i = 0; i < pending.length; i += BATCH_SIZE) {
      const batch = pending.slice(i, i + BATCH_SIZE);

      // Mark batch as uploading
      setFiles((prev) =>
        prev.map((f) =>
          batch.some((b) => b.id === f.id) ? { ...f, status: "uploading", progress: 10 } : f
        )
      );

      try {
        const batchFiles = batch.map((b) => b.file);
        // Simulate progress: jump to 50% while server processes
        setFiles((prev) =>
          prev.map((f) =>
            batch.some((b) => b.id === f.id) ? { ...f, progress: 50 } : f
          )
        );

        await uploadFiles(`/datasets/${datasetId}/upload`, batchFiles);

        setFiles((prev) =>
          prev.map((f) =>
            batch.some((b) => b.id === f.id)
              ? { ...f, status: "done", progress: 100 }
              : f
          )
        );
        doneCount += batch.length;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Upload failed";
        setFiles((prev) =>
          prev.map((f) =>
            batch.some((b) => b.id === f.id)
              ? { ...f, status: "error", progress: 0, errorMessage: message }
              : f
          )
        );
        errorCount += batch.length;
      }

      setUploadStats({ total: pending.length, done: doneCount, errors: errorCount });
    }

    setIsUploading(false);
    if (doneCount > 0) {
      onUploadComplete(doneCount);
    }
  }, [files, isUploading, datasetId, onUploadComplete]);

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------

  const pendingCount = files.filter((f) => f.status === "pending").length;
  const doneCount = files.filter((f) => f.status === "done").length;
  const errorCount = files.filter((f) => f.status === "error").length;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-3">
      {/* Drop area */}
      <div
        {...getRootProps()}
        className={cn(
          "relative flex flex-col items-center justify-center p-8 rounded-xl",
          "border-2 border-dashed transition-all duration-150 cursor-pointer",
          isDragActive
            ? "border-blue-500 bg-blue-500/10 scale-[1.01]"
            : "border-[var(--color-border)] hover:border-blue-500/50 hover:bg-[var(--color-surface)]"
        )}
      >
        <input {...getInputProps()} />
        <Upload
          className={cn(
            "w-8 h-8 mb-3 transition-colors",
            isDragActive ? "text-blue-400" : "text-[var(--color-text-muted)]"
          )}
        />
        <p className="text-sm font-medium text-[var(--color-text-secondary)]">
          {isDragActive ? "Drop images here" : "Drag & drop images, or click to browse"}
        </p>
        <p className="text-xs text-[var(--color-text-muted)] mt-1">
          JPG, PNG, TIFF, BMP — up to {maxFileSizeMb} MB each
        </p>
      </div>

      {/* Rejection messages */}
      {fileRejections.length > 0 && (
        <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20">
          <p className="text-xs text-red-400">
            {fileRejections.length} file(s) rejected:{" "}
            {fileRejections
              .slice(0, 3)
              .map((r) => `${r.file.name} (${r.errors.map((e) => e.message).join(", ")})`)
              .join("; ")}
            {fileRejections.length > 3 && ` …and ${fileRejections.length - 3} more`}
          </p>
        </div>
      )}

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1 scrollbar-thin">
          {files.map((item) => (
            <FileRow key={item.id} item={item} onRemove={() => removeFile(item.id)} />
          ))}
        </div>
      )}

      {/* Upload stats */}
      {uploadStats && (
        <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]">
          <span className="text-green-400 font-medium">
            ✓ {uploadStats.done} uploaded
          </span>
          {uploadStats.errors > 0 && (
            <span className="text-red-400 font-medium">
              ✗ {uploadStats.errors} failed
            </span>
          )}
          <span>{uploadStats.total} total</span>
        </div>
      )}

      {/* Action buttons */}
      {files.length > 0 && (
        <div className="flex items-center gap-2">
          {pendingCount > 0 && (
            <button
              onClick={handleUpload}
              disabled={isUploading}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                isUploading
                  ? "bg-blue-600/50 text-white/60 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-500 text-white"
              )}
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading…
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Upload {pendingCount} image{pendingCount !== 1 ? "s" : ""}
                </>
              )}
            </button>
          )}

          {doneCount > 0 && !isUploading && (
            <button
              onClick={clearDone}
              className="px-3 py-2 rounded-lg text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] border border-[var(--color-border)] hover:border-[var(--color-border-hover)] transition-colors"
            >
              Clear done ({doneCount})
            </button>
          )}

          {errorCount === 0 && pendingCount === 0 && !isUploading && (
            <span className="text-xs text-green-400 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              All uploads complete
            </span>
          )}
        </div>
      )}
    </div>
  );
}
