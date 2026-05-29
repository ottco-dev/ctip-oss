"use client";

/**
 * Interactive Workflow Guide — end-to-end CTIP onboarding.
 *
 * Steps:
 *   1  Image Collection        — microscope tips, file formats, resolution
 *   2  Dataset Creation        — import images, verify quality, set metadata
 *   3  Labeling                — Label Studio workflow, class guide, quality gates
 *   4  Training                — launch run, pick model, understand metrics
 *   5  Verification            — review mAP50, inspect detections, calibrate conf
 *   6  Benchmarking            — measure latency, FPS, VRAM, regression check
 *   7  Pipeline Builder        — assemble node graph, save shareable test link
 *
 * Each step has: description, tips, warnings, and a "Go there" button.
 */

import React, { useState } from "react";
import Link from "next/link";
import {
  Camera,
  Database,
  Tag,
  Cpu,
  CheckCircle2,
  BarChart3,
  Workflow,
  ChevronRight,
  ChevronLeft,
  Lightbulb,
  AlertTriangle,
  ExternalLink,
  Circle,
  CheckCircle,
  Info,
  ArrowRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Step definitions
// ---------------------------------------------------------------------------

interface Tip {
  type: "tip" | "warning" | "info";
  text: string;
}

interface Step {
  id: number;
  label: string;
  icon: React.ElementType;
  color: string;
  headline: string;
  summary: string;
  checklist: string[];
  tips: Tip[];
  link?: { href: string; label: string };
  details: React.ReactNode;
}

const STEPS: Step[] = [
  {
    id: 1,
    label: "Image Collection",
    icon: Camera,
    color: "#60a5fa",
    headline: "Capture high-quality microscopy images",
    summary:
      "Consistent, well-lit images are the single biggest factor in model accuracy. Bad images cannot be fixed by better training.",
    checklist: [
      "Use a digital microscope at 40× – 200× magnification",
      "Fix exposure and white balance — do NOT use auto-WB between shots",
      "Minimum 1280×1280 px per image (higher is better for tiled inference)",
      "Capture multiple focal planes per sample",
      "Label each image with: strain, harvest date, microscope ID, lighting preset",
      "Minimum 300 images across diverse strains for a useful dataset",
    ],
    tips: [
      {
        type: "tip",
        text: "Use a consistent light diffuser. Uneven lighting causes false amber in clear trichomes.",
      },
      {
        type: "tip",
        text: "Save as PNG or TIFF — never JPEG for training data (lossy compression degrades small features).",
      },
      {
        type: "warning",
        text: "Avoid motion blur: mount the microscope on a stable platform. Blurry trichomes teach the model to detect blur.",
      },
      {
        type: "info",
        text: "CTIP supports JPG/PNG/TIFF. For 4K images, tiled inference at runtime will handle them — no downscaling needed during data collection.",
      },
    ],
    link: { href: "/datasets", label: "Open Datasets" },
    details: (
      <div className="space-y-3 text-sm text-text-secondary">
        <p>
          The model learns exactly what you show it. A batch of 50 perfectly
          consistent images often outperforms 500 variable ones.
        </p>
        <div className="rounded-lg bg-panel border border-border p-3 font-mono text-xs space-y-1">
          <p className="text-text-muted">Recommended folder structure:</p>
          <p>raw_captures/</p>
          <p className="pl-4">strain_name/</p>
          <p className="pl-8">2025-05-28_001.tiff</p>
          <p className="pl-8">2025-05-28_002.tiff</p>
          <p className="pl-8">metadata.json</p>
        </div>
        <p>
          <strong className="text-text-primary">metadata.json</strong> fields to track:
          strain, harvest_date, microscope_model, magnification, lighting, operator.
        </p>
      </div>
    ),
  },

  {
    id: 2,
    label: "Dataset Creation",
    icon: Database,
    color: "#34d399",
    headline: "Import images and build a clean dataset",
    summary:
      "Use the Datasets page to create a named dataset, import images, and verify the distribution before labeling.",
    checklist: [
      "Create a new dataset in CTIP with a descriptive name (e.g. ctip-v3-multistrains)",
      "Import images via drag-and-drop or folder upload",
      "Run automatic quality check (blur detection, size validation)",
      "Review flagged images — remove or note problems",
      "Set train/val/test split (recommended: 70/20/10)",
      "Export metadata snapshot for reproducibility",
    ],
    tips: [
      {
        type: "tip",
        text: "Name datasets with version numbers from the start. You WILL create more than one.",
      },
      {
        type: "tip",
        text: "Keep a holdout test set that is NEVER seen during training or validation — use it only for final evaluation.",
      },
      {
        type: "warning",
        text: "Do NOT use the same images for training and validation. CTIP's split tool prevents this, but verify manually when combining datasets.",
      },
      {
        type: "info",
        text: "The 'Verify Dataset' button runs: class distribution check, image size histogram, empty label detection, and train/val leakage check.",
      },
    ],
    link: { href: "/datasets", label: "Open Datasets" },
    details: (
      <div className="space-y-3 text-sm text-text-secondary">
        <p>
          A healthy dataset has at minimum 3 samples per class per split. For
          trichome detection: Stalked, Sessile, Bulbous, Non-glandular.
        </p>
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-1.5 text-text-muted">Class</th>
              <th className="text-right py-1.5 text-text-muted">Min images</th>
              <th className="text-right py-1.5 text-text-muted">Target</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Stalked (capitate-stalked)", "50", "200+"],
              ["Sessile (capitate-sessile)", "40", "150+"],
              ["Bulbous", "20", "80+"],
              ["Non-glandular (hair)", "30", "100+"],
            ].map(([cls, min, tgt]) => (
              <tr key={cls} className="border-b border-border/50">
                <td className="py-1.5 text-text-secondary">{cls}</td>
                <td className="py-1.5 text-right text-orange-400 font-mono">{min}</td>
                <td className="py-1.5 text-right text-emerald-400 font-mono">{tgt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ),
  },

  {
    id: 3,
    label: "Labeling",
    icon: Tag,
    color: "#a78bfa",
    headline: "Annotate trichomes in Label Studio",
    summary:
      "Open the Annotation tab, connect to Label Studio, and systematically label every trichome with bounding boxes.",
    checklist: [
      "Connect to Label Studio in the Annotation tab",
      "Create or select a project with the CTIP bounding box template",
      "Select your dataset in the Label Studio tab",
      "Enable VLM auto-annotation for a first-pass pre-label (saves ~60% of manual work)",
      "Review ALL auto-labels — never accept without checking",
      "Use consensus labeling for ambiguous trichomes (label, ask second reviewer)",
      "Achieve >95% inter-annotator agreement before training",
      "Export to YOLO format and verify class distribution",
    ],
    tips: [
      {
        type: "tip",
        text: "Start labeling with easy, clear images first. This trains your eye and creates a consistent reference before tackling edge cases.",
      },
      {
        type: "tip",
        text: "Tight boxes (touching the trichome head edge) consistently outperform loose boxes in YOLO training.",
      },
      {
        type: "warning",
        text: "VLM auto-labels MUST be human-reviewed before entering the training dataset. This is a hard system constraint.",
      },
      {
        type: "warning",
        text: "Do NOT label trichomes that are partially cut off by the image edge — YOLO will learn to detect partial objects inconsistently.",
      },
      {
        type: "info",
        text: "Use 'Active Learning' to prioritize the most informative images to label next — saves time when you have 1000+ unlabeled images.",
      },
    ],
    link: { href: "/annotation", label: "Open Annotation" },
    details: (
      <div className="space-y-3 text-sm text-text-secondary">
        <p>
          <strong className="text-text-primary">Class guide:</strong>
        </p>
        <ul className="space-y-1.5 text-xs">
          <li>
            <span className="text-blue-400 font-medium">Stalked</span> — large
            glandular head on a visible stalk. Most common in mature flower.
          </li>
          <li>
            <span className="text-purple-400 font-medium">Sessile</span> — smaller
            head, no visible stalk. Common on sugar leaves.
          </li>
          <li>
            <span className="text-orange-400 font-medium">Bulbous</span> — tiny,
            round, no stalk. Often at the base of larger trichomes.
          </li>
          <li>
            <span className="text-gray-400 font-medium">Non-glandular</span> — hair
            structures, no head. Long thin fibers.
          </li>
        </ul>
        <p className="text-xs text-text-muted">
          When in doubt between Sessile and Stalked: if you can see a distinct stalk
          segment, it is Stalked.
        </p>
      </div>
    ),
  },

  {
    id: 4,
    label: "Training",
    icon: Cpu,
    color: "#f59e0b",
    headline: "Launch a training run from the Training page",
    summary:
      "Select your dataset, model variant, and hyperparameters. CTIP handles the rest: MLflow logging, auto-registration on completion.",
    checklist: [
      "Go to Training → Runs tab",
      "Select dataset (use bare name: ctip-v3-multistrains)",
      "Pick model variant: yolo11s for RTX 4060 (best speed/accuracy balance)",
      "Set epochs: 100–200 (use early stopping patience=50)",
      "Batch size 4 for 8 GB VRAM, 2 for 6 GB VRAM",
      "Enable AMP (mixed precision) — saves ~1 GB VRAM",
      "Hit Start Training, watch live metrics in the Metrics chart",
      "After completion, verify model appears in Models registry",
    ],
    tips: [
      {
        type: "tip",
        text: "Always create an Experiment before training. Groups related runs for comparison.",
      },
      {
        type: "tip",
        text: "Use seed=42 for all runs to compare hyperparameter changes fairly.",
      },
      {
        type: "warning",
        text: "Do NOT close the browser during training — the backend continues, but you lose the live log. Reconnect via Training page.",
      },
      {
        type: "info",
        text: "yolo11n: fastest (6 ms/img) but lower mAP. yolo11m: highest mAP but 2.5× VRAM. Start with yolo11s.",
      },
    ],
    link: { href: "/training", label: "Open Training" },
    details: (
      <div className="space-y-3 text-sm text-text-secondary">
        <p>
          <strong className="text-text-primary">When to stop training:</strong>
        </p>
        <ul className="space-y-1 text-xs">
          <li>✓ mAP50 &gt; 0.85 AND validation loss has plateaued for 30+ epochs</li>
          <li>✓ Train/val loss gap is small (no overfitting)</li>
          <li>✗ Never stop before epoch 50 — YOLO needs warmup time</li>
        </ul>
        <p className="mt-2">
          <strong className="text-text-primary">Interpreting the metrics chart:</strong>
        </p>
        <table className="w-full text-xs border-collapse">
          <tbody>
            {[
              ["box_loss", "Detection localization — should decrease steadily"],
              ["cls_loss", "Classification — should approach 0.1 for clean datasets"],
              ["mAP50", "Main metric — target ≥ 0.85"],
              ["mAP50-95", "Strict metric — target ≥ 0.60"],
            ].map(([m, d]) => (
              <tr key={m} className="border-b border-border/50">
                <td className="py-1.5 font-mono text-accent pr-4">{m}</td>
                <td className="py-1.5 text-text-muted">{d}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ),
  },

  {
    id: 5,
    label: "Verification",
    icon: CheckCircle2,
    color: "#22c55e",
    headline: "Verify the trained model on unseen test data",
    summary:
      "Use the Detection Workbench to run the new model on your holdout test images and confirm the metrics match training logs.",
    checklist: [
      "Go to Inference → Workbench tab",
      "Select your trained model from the registry dropdown",
      "Upload 5–10 test images (images the model has NEVER seen)",
      "Check detection count matches manual expectation",
      "Verify confidence scores look reasonable (0.5–0.95 for clear cases)",
      "Try confidence threshold 0.35 → adjust if too many false positives",
      "Check all 4 classes get detected (missing class = labeling gap)",
      "If mAP looks wrong: run Evaluation → Calibration tab with predictions",
    ],
    tips: [
      {
        type: "tip",
        text: "Use the Pipeline Builder for systematic testing: Image Input → Model → Stats Output. Connect multiple test images in one go.",
      },
      {
        type: "tip",
        text: "Confidence threshold 0.35 is the default. For high-precision applications (research counting), raise to 0.5.",
      },
      {
        type: "warning",
        text: "A model that scores mAP50=0.95 on training data but 0.60 on test images is overfitting. Return to labeling step and add more diverse images.",
      },
      {
        type: "info",
        text: "The Evaluation → Calibration tab lets you compute ECE (Expected Calibration Error) to check if confidence scores are meaningful.",
      },
    ],
    link: { href: "/inference", label: "Open Inference" },
    details: (
      <div className="space-y-3 text-sm text-text-secondary">
        <p>
          <strong className="text-text-primary">Common verification failures and fixes:</strong>
        </p>
        <ul className="space-y-2 text-xs">
          <li>
            <span className="text-red-400">Missing detections</span> — lower confidence
            threshold, or add more labeled examples of that trichome type
          </li>
          <li>
            <span className="text-orange-400">Too many false positives</span> — raise
            threshold or add hard-negative examples to training data
          </li>
          <li>
            <span className="text-yellow-400">Wrong class assignment</span> — review
            label consistency; Sessile/Stalked confusion is most common
          </li>
          <li>
            <span className="text-blue-400">Works on one strain, fails on another</span>{" "}
            — your training data is not diverse enough; add images from the failing strain
          </li>
        </ul>
      </div>
    ),
  },

  {
    id: 6,
    label: "Benchmarking",
    icon: BarChart3,
    color: "#f472b6",
    headline: "Measure latency, FPS, and VRAM usage",
    summary:
      "Run the Benchmarks tab with your test images to confirm the model meets your latency requirements on the target hardware.",
    checklist: [
      "Go to Evaluation → Benchmarks tab",
      "Upload 10–20 representative images",
      "Run benchmark with conf_threshold=0.35",
      "Record mean FPS and mean latency (ms/image)",
      "Compare to reference benchmarks table (yolo11s target: ~75 FPS on RTX 4060)",
      "If latency is too high: try yolo11n or enable half precision",
      "For 4K images: test with use_tiled=True and compare",
      "Save benchmark result for regression tracking",
    ],
    tips: [
      {
        type: "tip",
        text: "Run the benchmark 3 times and average — first run is always slower due to CUDA graph compilation.",
      },
      {
        type: "tip",
        text: "For production use, run benchmarks with the same images at every model version update to catch regressions.",
      },
      {
        type: "warning",
        text: "Browser-measured latency includes HTTP round-trip time (~5 ms overhead). For pure model latency, check the inference_time_ms field in API responses.",
      },
      {
        type: "info",
        text: "Reference: yolo11s at imgsz=1280 → ~13 ms/img, 75 FPS. Tiled 4K → ~150 ms/img, 7 FPS. These are RTX 4060 FP16 numbers.",
      },
    ],
    link: { href: "/evaluation?tab=benchmarks", label: "Open Benchmarks" },
    details: (
      <div className="space-y-3 text-sm text-text-secondary">
        <p>
          <strong className="text-text-primary">Acceptable latency targets:</strong>
        </p>
        <table className="w-full text-xs border-collapse">
          <tbody>
            {[
              ["Live preview (interactive)", "< 100 ms", "≥ 10 FPS"],
              ["Batch processing (research)", "< 500 ms", "≥ 2 FPS"],
              ["4K tiled inference", "< 2 s", "—"],
            ].map(([use, lat, fps]) => (
              <tr key={use} className="border-b border-border/50">
                <td className="py-1.5 text-text-secondary">{use}</td>
                <td className="py-1.5 text-right text-emerald-400 font-mono">{lat}</td>
                <td className="py-1.5 text-right text-blue-400 font-mono">{fps}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ),
  },

  {
    id: 7,
    label: "Pipeline Builder",
    icon: Workflow,
    color: "#e879f9",
    headline: "Build a shareable test pipeline with drag-and-drop nodes",
    summary:
      "Use the Inference → Pipeline Builder to visually connect Image Input → Model → Filter → Output nodes into a reproducible test setup.",
    checklist: [
      "Go to Inference → Pipeline Builder tab",
      "Drag an Image Input node from the left palette",
      "Drag a Model node, select your trained model from the dropdown",
      "Connect Image Input output → Model input (drag the dot)",
      "Drag a Detection Output node, connect Model output → Detection Output",
      "Optionally add a Filter node between Model and Output",
      "Upload a test image in the Image Input node",
      "Click Run — detections appear in the output node",
      "Click Save, then Share to copy a shareable URL to the pipeline",
    ],
    tips: [
      {
        type: "tip",
        text: "Add a Stats Output node alongside Detection Output to see class distribution at a glance.",
      },
      {
        type: "tip",
        text: "Save multiple pipeline configurations for different use cases: high-precision (conf=0.7), high-recall (conf=0.2), 4K tiled.",
      },
      {
        type: "info",
        text: "Saved pipelines are stored in the database with UUIDs. Share the URL with collaborators — they can load and run the exact same configuration.",
      },
    ],
    link: { href: "/inference?tab=pipeline", label: "Open Pipeline Builder" },
    details: (
      <div className="space-y-3 text-sm text-text-secondary">
        <p>
          <strong className="text-text-primary">Available node types:</strong>
        </p>
        <div className="space-y-1.5 text-xs">
          {[
            { color: "#60a5fa", name: "Image Input", desc: "Upload zone — entry point for images" },
            { color: "#a78bfa", name: "Model", desc: "Detection with conf/IoU/tiled controls" },
            { color: "#34d399", name: "Filter", desc: "Confidence gate + class whitelist" },
            { color: "#f472b6", name: "Detection Output", desc: "Annotated image with bounding boxes" },
            { color: "#fb923c", name: "Stats Output", desc: "Per-class count + avg confidence chart" },
          ].map(({ color, name, desc }) => (
            <div key={name} className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
              <span className="font-medium" style={{ color }}>{name}</span>
              <span className="text-text-muted">— {desc}</span>
            </div>
          ))}
        </div>
      </div>
    ),
  },
];

// ---------------------------------------------------------------------------
// Step indicator
// ---------------------------------------------------------------------------

function StepIndicator({
  steps,
  current,
  completed,
  onSelect,
}: {
  steps: Step[];
  current: number;
  completed: Set<number>;
  onSelect: (id: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      {steps.map((step, i) => {
        const Icon = step.icon;
        const isActive = step.id === current;
        const isDone = completed.has(step.id);

        return (
          <button
            key={step.id}
            onClick={() => onSelect(step.id)}
            className={cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all",
              isActive
                ? "bg-accent/10 border border-accent/30"
                : "hover:bg-panel border border-transparent"
            )}
          >
            <div className="relative shrink-0">
              {isDone ? (
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              ) : isActive ? (
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold"
                  style={{ background: step.color, color: "#000" }}
                >
                  {step.id}
                </div>
              ) : (
                <Circle
                  className="w-5 h-5"
                  style={{ color: isDone ? "#22c55e" : "#374151" }}
                />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p
                className={cn(
                  "text-xs font-medium truncate",
                  isActive ? "text-text-primary" : "text-text-secondary"
                )}
              >
                {step.label}
              </p>
            </div>
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tip block
// ---------------------------------------------------------------------------

function TipBlock({ tip }: { tip: Tip }) {
  const config = {
    tip: { icon: Lightbulb, color: "text-yellow-400", bg: "bg-yellow-400/10 border-yellow-400/20" },
    warning: { icon: AlertTriangle, color: "text-orange-400", bg: "bg-orange-400/10 border-orange-400/20" },
    info: { icon: Info, color: "text-blue-400", bg: "bg-blue-400/10 border-blue-400/20" },
  }[tip.type];

  const TipIcon = config.icon;

  return (
    <div className={cn("flex gap-2.5 p-3 rounded-lg border text-xs", config.bg)}>
      <TipIcon className={cn("w-3.5 h-3.5 shrink-0 mt-0.5", config.color)} />
      <span className="text-text-secondary">{tip.text}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function GuidePage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [completed, setCompleted] = useState<Set<number>>(new Set());

  const step = STEPS.find((s) => s.id === currentStep)!;
  const Icon = step.icon;

  function markDone(id: number) {
    setCompleted((prev) => new Set([...prev, id]));
  }

  function goNext() {
    markDone(currentStep);
    if (currentStep < STEPS.length) setCurrentStep(currentStep + 1);
  }

  function goPrev() {
    if (currentStep > 1) setCurrentStep(currentStep - 1);
  }

  const progress = Math.round((completed.size / STEPS.length) * 100);

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Left sidebar ──────────────────────────────────────────────────── */}
      <div className="w-52 shrink-0 border-r border-border bg-surface flex flex-col overflow-hidden">
        <div className="px-4 py-4 border-b border-border">
          <h2 className="text-sm font-semibold text-text-primary">Workflow Guide</h2>
          <p className="text-xs text-text-muted mt-0.5">
            {completed.size}/{STEPS.length} steps complete
          </p>
          <div className="mt-2 h-1 bg-panel rounded-full overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-2">
          <StepIndicator
            steps={STEPS}
            current={currentStep}
            completed={completed}
            onSelect={setCurrentStep}
          />
        </div>

        {completed.size === STEPS.length && (
          <div className="px-4 py-3 border-t border-border">
            <div className="flex items-center gap-2 text-emerald-400 text-xs font-medium">
              <CheckCircle className="w-4 h-4" />
              Pipeline complete!
            </div>
          </div>
        )}
      </div>

      {/* ── Main content ──────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Step header */}
        <div className="border-b border-border px-6 py-4 flex items-center gap-4 shrink-0">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: `${step.color}18`, border: `1px solid ${step.color}30` }}
          >
            <Icon className="w-5 h-5" style={{ color: step.color }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span
                className="text-xs font-medium px-1.5 py-0.5 rounded"
                style={{ background: `${step.color}20`, color: step.color }}
              >
                Step {step.id} of {STEPS.length}
              </span>
              {completed.has(step.id) && (
                <span className="text-xs text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Completed
                </span>
              )}
            </div>
            <h1 className="text-base font-semibold text-text-primary mt-0.5">
              {step.headline}
            </h1>
          </div>
          {step.link && (
            <Link
              href={step.link.href}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent hover:bg-accent/80 text-background transition-colors shrink-0"
            >
              {step.link.label}
              <ExternalLink className="w-3 h-3" />
            </Link>
          )}
        </div>

        {/* Step body */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-6 py-6 space-y-6">
            {/* Summary */}
            <p className="text-sm text-text-secondary leading-relaxed border-l-2 pl-4" style={{ borderColor: step.color }}>
              {step.summary}
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Checklist */}
              <div>
                <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
                  Checklist
                </h3>
                <ChecklistSection items={step.checklist} stepId={step.id} />
              </div>

              {/* Tips */}
              <div>
                <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
                  Tips & Warnings
                </h3>
                <div className="space-y-2">
                  {step.tips.map((tip, i) => (
                    <TipBlock key={i} tip={tip} />
                  ))}
                </div>
              </div>
            </div>

            {/* Detail panel */}
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
                Details
              </h3>
              {step.details}
            </div>
          </div>
        </div>

        {/* Navigation footer */}
        <div className="border-t border-border px-6 py-3 flex items-center gap-3 shrink-0">
          <button
            onClick={goPrev}
            disabled={currentStep === 1}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-text-muted hover:text-text-primary hover:bg-panel transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </button>

          <button
            onClick={() => markDone(currentStep)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors",
              completed.has(step.id)
                ? "text-emerald-400 bg-emerald-400/10 border border-emerald-400/20"
                : "text-text-muted hover:text-text-primary hover:bg-panel border border-border"
            )}
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            {completed.has(step.id) ? "Done" : "Mark done"}
          </button>

          <div className="flex-1" />

          {step.link && (
            <Link
              href={step.link.href}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-accent hover:text-accent/80 transition-colors"
            >
              <ArrowRight className="w-3.5 h-3.5" />
              {step.link.label}
            </Link>
          )}

          <button
            onClick={goNext}
            disabled={currentStep === STEPS.length}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent hover:bg-accent/80 text-background transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next step
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Interactive checklist with persistent state per step
// ---------------------------------------------------------------------------

function ChecklistSection({ items, stepId }: { items: string[]; stepId: number }) {
  const [checked, setChecked] = useState<Set<number>>(new Set());

  function toggle(i: number) {
    setChecked((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  }

  return (
    <div className="space-y-1.5">
      {items.map((item, i) => (
        <label
          key={i}
          className="flex items-start gap-2.5 cursor-pointer group"
          onClick={() => toggle(i)}
        >
          <div
            className={cn(
              "w-4 h-4 mt-0.5 shrink-0 rounded border transition-all",
              checked.has(i)
                ? "bg-emerald-500 border-emerald-500"
                : "border-border group-hover:border-text-muted"
            )}
          >
            {checked.has(i) && (
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 16 16">
                <path
                  stroke="currentColor"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 8l3.5 3.5L13 5"
                />
              </svg>
            )}
          </div>
          <span
            className={cn(
              "text-xs leading-relaxed transition-colors",
              checked.has(i) ? "text-text-muted line-through" : "text-text-secondary"
            )}
          >
            {item}
          </span>
        </label>
      ))}
      {checked.size > 0 && (
        <p className="text-[10px] text-text-muted pt-1">
          {checked.size}/{items.length} checked
        </p>
      )}
    </div>
  );
}
