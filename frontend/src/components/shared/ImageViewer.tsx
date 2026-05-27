'use client';

/**
 * ImageViewer — Zoomable, pannable image with annotation overlays.
 *
 * Renders:
 *   • Bounding boxes (detection)
 *   • Polygon masks (segmentation)
 *   • Labels with confidence scores
 *   • Maturity stage colours
 *
 * Usage:
 *   <ImageViewer
 *     src="/path/to/image.jpg"
 *     annotations={detections}
 *     showLabels
 *     showConfidence
 *     onAnnotationClick={(id) => ...}
 *   />
 */

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  WheelEvent,
  PointerEvent as ReactPointerEvent,
} from 'react';
import { ZoomIn, ZoomOut, Maximize2, RotateCcw, Eye, EyeOff } from 'lucide-react';
import { cn } from '@/lib/utils';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface AnnotationBox {
  id: string;
  /** Normalised [0,1] or absolute pixel coords — set coordsType */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label?: string;
  confidence?: number;
  color?: string;
  /** Polygon mask points as flat [x,y,x,y,...] in same coord space */
  mask?: number[];
}

export type CoordsType = 'pixel' | 'normalized';

interface ViewerTransform {
  scale: number;
  tx: number; // translation X in CSS pixels
  ty: number; // translation Y in CSS pixels
}

interface ImageViewerProps {
  src: string;
  alt?: string;
  annotations?: AnnotationBox[];
  coordsType?: CoordsType;
  showLabels?: boolean;
  showConfidence?: boolean;
  showMasks?: boolean;
  /** Hide annotation overlay entirely */
  overlayVisible?: boolean;
  className?: string;
  onAnnotationClick?: (id: string) => void;
  selectedId?: string | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour helpers
// ─────────────────────────────────────────────────────────────────────────────

const PALETTE = [
  '#22d3ee', // capitate_stalked / cyan
  '#34d399', // capitate_sessile / green
  '#a78bfa', // bulbous / purple
  '#fb923c', // non_glandular / orange
  '#f87171', // misc / red
  '#60a5fa', // misc / blue
  '#facc15', // misc / yellow
];

const LABEL_COLORS: Record<string, string> = {
  capitate_stalked: '#22d3ee',
  capitate_sessile: '#34d399',
  bulbous: '#a78bfa',
  non_glandular: '#fb923c',
  clear: '#60a5fa',
  cloudy: '#e5e7eb',
  amber: '#f59e0b',
  degraded: '#a16207',
};

function annotationColor(ann: AnnotationBox, idx: number): string {
  if (ann.color) return ann.color;
  const label = ann.label?.toLowerCase() ?? '';
  return LABEL_COLORS[label] ?? PALETTE[idx % PALETTE.length];
}

function confidenceAlpha(conf: number | undefined): number {
  if (conf == null) return 0.7;
  return 0.4 + conf * 0.5; // 0.4 … 0.9
}

// ─────────────────────────────────────────────────────────────────────────────
// Transform helpers
// ─────────────────────────────────────────────────────────────────────────────

const MIN_SCALE = 0.1;
const MAX_SCALE = 10;
const ZOOM_STEP = 0.15;

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export function ImageViewer({
  src,
  alt = 'microscopy image',
  annotations = [],
  coordsType = 'pixel',
  showLabels = true,
  showConfidence = true,
  showMasks = true,
  overlayVisible = true,
  className,
  onAnnotationClick,
  selectedId,
}: ImageViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const [transform, setTransform] = useState<ViewerTransform>({ scale: 1, tx: 0, ty: 0 });
  const [imageLoaded, setImageLoaded] = useState(false);
  const [naturalSize, setNaturalSize] = useState({ w: 1, h: 1 });
  const [showOverlay, setShowOverlay] = useState(overlayVisible);

  // Panning state
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0, tx: 0, ty: 0 });

  // ── Reset on new image ──────────────────────────────────────────────────

  useEffect(() => {
    setImageLoaded(false);
    setTransform({ scale: 1, tx: 0, ty: 0 });
  }, [src]);

  const handleImageLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    setImageLoaded(true);
    // Fit to container on first load
    const container = containerRef.current;
    if (container) {
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight, 1);
      setTransform({ scale, tx: 0, ty: 0 });
    }
  }, []);

  // ── Zoom ────────────────────────────────────────────────────────────────

  const zoom = useCallback((delta: number, cx?: number, cy?: number) => {
    setTransform((prev) => {
      const next = clamp(prev.scale + delta * prev.scale, MIN_SCALE, MAX_SCALE);
      const ratio = next / prev.scale;
      // Zoom towards cursor if provided
      const ox = cx ?? 0;
      const oy = cy ?? 0;
      return {
        scale: next,
        tx: ox - ratio * (ox - prev.tx),
        ty: oy - ratio * (oy - prev.ty),
      };
    });
  }, []);

  const resetView = useCallback(() => {
    const img = imgRef.current;
    const container = containerRef.current;
    if (!img || !container) return;
    const scale = Math.min(
      container.clientWidth / naturalSize.w,
      container.clientHeight / naturalSize.h,
      1,
    );
    setTransform({ scale, tx: 0, ty: 0 });
  }, [naturalSize]);

  // ── Mouse wheel zoom ─────────────────────────────────────────────────────

  const onWheel = useCallback(
    (e: WheelEvent<HTMLDivElement>) => {
      e.preventDefault();
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      zoom(e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP, cx, cy);
    },
    [zoom],
  );

  // ── Pan (pointer events) ─────────────────────────────────────────────────

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      isPanning.current = true;
      panStart.current = {
        x: e.clientX,
        y: e.clientY,
        tx: transform.tx,
        ty: transform.ty,
      };
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [transform],
  );

  const onPointerMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    if (!isPanning.current) return;
    const dx = e.clientX - panStart.current.x;
    const dy = e.clientY - panStart.current.y;
    setTransform((prev) => ({
      ...prev,
      tx: panStart.current.tx + dx,
      ty: panStart.current.ty + dy,
    }));
  }, []);

  const onPointerUp = useCallback(() => {
    isPanning.current = false;
  }, []);

  // ── SVG overlay coordinate conversion ───────────────────────────────────

  /** Convert annotation coord → SVG pixel coord in natural image space */
  const toSvgCoord = useCallback(
    (x: number, y: number): [number, number] => {
      if (coordsType === 'normalized') {
        return [x * naturalSize.w, y * naturalSize.h];
      }
      return [x, y];
    },
    [coordsType, naturalSize],
  );

  // ── CSS transform string ────────────────────────────────────────────────

  const transformCss = `translate(${transform.tx}px, ${transform.ty}px) scale(${transform.scale})`;

  return (
    <div className={cn('relative w-full h-full overflow-hidden bg-[#0d1117] select-none', className)}>
      {/* Zoom controls */}
      <div className="absolute top-2 right-2 z-20 flex flex-col gap-1">
        <button
          onClick={() => zoom(ZOOM_STEP)}
          className="w-7 h-7 rounded bg-black/60 border border-white/10 flex items-center justify-center text-white/60 hover:text-white hover:bg-black/80 transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => zoom(-ZOOM_STEP)}
          className="w-7 h-7 rounded bg-black/60 border border-white/10 flex items-center justify-center text-white/60 hover:text-white hover:bg-black/80 transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={resetView}
          className="w-7 h-7 rounded bg-black/60 border border-white/10 flex items-center justify-center text-white/60 hover:text-white hover:bg-black/80 transition-colors"
          title="Reset view"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
        {annotations.length > 0 && (
          <button
            onClick={() => setShowOverlay((v) => !v)}
            className="w-7 h-7 rounded bg-black/60 border border-white/10 flex items-center justify-center text-white/60 hover:text-white hover:bg-black/80 transition-colors"
            title={showOverlay ? 'Hide overlay' : 'Show overlay'}
          >
            {showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>

      {/* Scale indicator */}
      <div className="absolute bottom-2 left-2 z-20 text-[10px] text-white/40 font-mono bg-black/40 px-1.5 py-0.5 rounded">
        {(transform.scale * 100).toFixed(0)}%
      </div>

      {/* Detection count */}
      {annotations.length > 0 && (
        <div className="absolute bottom-2 right-2 z-20 text-[10px] text-white/40 font-mono bg-black/40 px-1.5 py-0.5 rounded">
          {annotations.length} detection{annotations.length !== 1 ? 's' : ''}
        </div>
      )}

      {/* Pannable / zoomable container */}
      <div
        ref={containerRef}
        className="w-full h-full flex items-center justify-center cursor-grab active:cursor-grabbing"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <div
          style={{ transform: transformCss, transformOrigin: 'center center', position: 'relative' }}
        >
          {/* Image */}
          <img
            ref={imgRef}
            src={src}
            alt={alt}
            onLoad={handleImageLoad}
            draggable={false}
            style={{
              display: 'block',
              maxWidth: 'none',
              imageRendering: transform.scale > 2 ? 'pixelated' : 'auto',
            }}
          />

          {/* SVG overlay — same dimensions as natural image */}
          {imageLoaded && showOverlay && annotations.length > 0 && (
            <svg
              ref={svgRef}
              width={naturalSize.w}
              height={naturalSize.h}
              viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                pointerEvents: 'none',
              }}
            >
              {annotations.map((ann, idx) => {
                const [sx1, sy1] = toSvgCoord(ann.x1, ann.y1);
                const [sx2, sy2] = toSvgCoord(ann.x2, ann.y2);
                const color = annotationColor(ann, idx);
                const alpha = confidenceAlpha(ann.confidence);
                const isSelected = ann.id === selectedId;
                const boxW = sx2 - sx1;
                const boxH = sy2 - sy1;

                const labelText =
                  showLabels && ann.label
                    ? showConfidence && ann.confidence != null
                      ? `${ann.label} ${(ann.confidence * 100).toFixed(0)}%`
                      : ann.label
                    : null;

                const strokeW = isSelected ? 2.5 : 1.5;

                return (
                  <g key={ann.id} style={{ pointerEvents: 'all', cursor: onAnnotationClick ? 'pointer' : 'default' }}
                    onClick={() => onAnnotationClick?.(ann.id)}>

                    {/* Mask polygon */}
                    {showMasks && ann.mask && ann.mask.length >= 6 && (
                      <polygon
                        points={(() => {
                          const pts: string[] = [];
                          for (let i = 0; i < ann.mask.length - 1; i += 2) {
                            const [px, py] = toSvgCoord(ann.mask[i], ann.mask[i + 1]);
                            pts.push(`${px},${py}`);
                          }
                          return pts.join(' ');
                        })()}
                        fill={color}
                        fillOpacity={0.18}
                        stroke={color}
                        strokeWidth={1}
                        strokeOpacity={0.5}
                      />
                    )}

                    {/* Bounding box */}
                    <rect
                      x={sx1}
                      y={sy1}
                      width={boxW}
                      height={boxH}
                      fill="none"
                      stroke={color}
                      strokeWidth={strokeW}
                      strokeOpacity={alpha}
                      rx={2}
                    />

                    {/* Selection glow */}
                    {isSelected && (
                      <rect
                        x={sx1 - 2}
                        y={sy1 - 2}
                        width={boxW + 4}
                        height={boxH + 4}
                        fill="none"
                        stroke={color}
                        strokeWidth={1}
                        strokeOpacity={0.3}
                        rx={3}
                      />
                    )}

                    {/* Label pill */}
                    {labelText && (
                      <g>
                        <rect
                          x={sx1}
                          y={sy1 - 16}
                          width={labelText.length * 6.2 + 8}
                          height={15}
                          fill={color}
                          fillOpacity={0.85}
                          rx={2}
                        />
                        <text
                          x={sx1 + 4}
                          y={sy1 - 5}
                          fontSize={10}
                          fill="white"
                          fontFamily="monospace"
                          fontWeight="500"
                        >
                          {labelText}
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Mini thumbnail variant (no controls)
// ─────────────────────────────────────────────────────────────────────────────

export function ImageThumbnail({
  src,
  alt,
  annotations = [],
  coordsType = 'pixel',
  className,
}: Pick<ImageViewerProps, 'src' | 'alt' | 'annotations' | 'coordsType' | 'className'>) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [size, setSize] = useState({ w: 1, h: 1 });
  const [loaded, setLoaded] = useState(false);

  const onLoad = () => {
    const img = imgRef.current;
    if (img) {
      setSize({ w: img.naturalWidth, h: img.naturalHeight });
      setLoaded(true);
    }
  };

  return (
    <div className={cn('relative overflow-hidden', className)}>
      <img
        ref={imgRef}
        src={src}
        alt={alt ?? ''}
        onLoad={onLoad}
        className="w-full h-full object-cover"
        draggable={false}
      />
      {loaded && annotations.length > 0 && (
        <svg
          width="100%"
          height="100%"
          viewBox={`0 0 ${size.w} ${size.h}`}
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        >
          {annotations.map((ann, idx) => {
            const toSvg = (x: number, y: number): [number, number] =>
              coordsType === 'normalized' ? [x * size.w, y * size.h] : [x, y];
            const [x1, y1] = toSvg(ann.x1, ann.y1);
            const [x2, y2] = toSvg(ann.x2, ann.y2);
            const color = annotationColor(ann, idx);
            return (
              <rect
                key={ann.id}
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill="none"
                stroke={color}
                strokeWidth={2}
                rx={1}
              />
            );
          })}
        </svg>
      )}
    </div>
  );
}
