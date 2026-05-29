# CTIP — Trade Show Cheat Sheet
### Cannabis Trichome Intelligence Platform · MaryJane Messe

---

## 🎯 ONE-LINER (say this first)

> **"We built an open-source AI platform that uses a microscope and computer vision
> to analyze cannabis trichomes — helping growers determine the optimal harvest window
> based on optical maturity, not guesswork."**

---

## 🃏 CARD 1 — The Problem

- Growers look at trichomes under a microscope by **eye**
- Subjective, inconsistent, time-consuming
- "Is it cloudy? Is it amber? How much?" → nobody agrees
- No data, no history, no reproducibility
- **→ We automate and quantify what the grower already does manually**

---

## 🃏 CARD 2 — The Solution (what it does)

| Step | What happens |
|---|---|
| 📷 Capture | Plug in USB microscope or use video |
| 🔍 Detect | YOLO AI finds every trichome in the image |
| 🧬 Classify | Morphology: stalked / sessile / bulbous / non-glandular |
| 🎨 Maturity | Color+texture analysis: **clear → cloudy → amber** |
| 📊 Report | PDF/JSON export, historical trends, harvest recommendation |

**Key point:** Optical analysis only — we describe what we see, not what's inside.

---

## 🃏 CARD 3 — The Science

- **Trichome types:**
  - Capitate-Stalked → the big ones, primary target
  - Capitate-Sessile → smaller, sessile
  - Bulbous → tiny, all over the plant
  - Non-Glandular → hair-like, no resin

- **Maturity stages (optical):**
  - 🔵 **Clear** — immature, still developing
  - ⚪ **Cloudy** — peak maturity window
  - 🟡 **Amber** — oxidizing, degradation beginning

- **We measure:** ratio of cloudy:amber:clear across ALL detected trichomes
  → gives a **population-level harvest score**, not a single sample

---

## 🃏 CARD 4 — Why It's Real Engineering

- **1,650 automated tests** — everything verified
- **YOLO v11** detection model (tile-based, 1280px, handles macro microscopy)
- **SAM 2** instance segmentation for precise masks
- **Kalman filter tracker** follows individual trichomes across video frames
- **Calibration pipeline** converts pixels → micrometers (µm)
- Full **REST API** — can connect to any external system
- Runs on a **single RTX 4060 (8 GB)** — no datacenter needed

---

## 🃏 CARD 5 — The Stack (for tech people)

```
Frontend:   Next.js 14 · TypeScript · React Query · Tailwind
Backend:    FastAPI · SQLite/PostgreSQL · asyncio
ML:         YOLO v11s · SAM 2 tiny · EfficientNet-B0
VLM:        Moondream · Florence-2 · Qwen2-VL (local, 4-bit)
            + OpenAI / Anthropic / Google / Groq / Together (cloud)
Tracking:   SORT (Kalman + Hungarian assignment)
Infra:      Docker Compose · Nginx · MLflow · Label Studio · CVAT
CI:         GitHub Actions · 1650 pytest · TypeScript strict
```

---

## 🃏 CARD 6 — Key Features (demo flow)

1. **Upload image / video** → instant trichome detection with bounding boxes
2. **Batch inference** → 3× throughput via dynamic batching queue
3. **Maturity heatmap** → color-coded distribution per image
4. **Video tracking** → watch trichomes age frame-by-frame (time-lapse)
5. **AI report narrative** → Ollama local LLM writes a scientific harvest summary
6. **Annotation workflow** → VLM suggests labels → human reviews → training data
7. **Live dashboard** → GPU stats, active jobs, real-time training metrics

---

## 🃏 CARD 7 — Human-in-the-Loop (HITL)

**This is important — say it clearly:**

> "The AI proposes labels. A human must approve every annotation before
> it enters training data. This is enforced in code — not a policy."

- VLM generates candidate labels → pushed to **review queue**
- Reviewer sees image + bounding boxes + confidence scores
- **Approve / Reject** with keyboard shortcuts (A / R)
- Only approved labels reach training datasets
- **Protects scientific integrity**

---

## 🃏 CARD 8 — What We Are NOT Saying

❌ We do NOT predict THC/CBD concentration  
❌ We do NOT replace lab testing (HPLC, GC-MS)  
❌ We do NOT make legal/medical claims  

✅ We describe **optical appearance** of trichome structures  
✅ Optical maturity correlates with **what experienced growers already observe**  
✅ We make that observation **consistent, quantified, reproducible**

---

## 🃏 CARD 9 — Who Is This For?

| Audience | Value |
|---|---|
| **Small craft grower** | Stop guessing harvest time; track multiple plants |
| **Commercial grow op** | Consistent quality across rooms/batches; data trail |
| **Breeder** | Document phenotype maturity profiles objectively |
| **Research lab** | Reproducible optical maturity data, YOLO/ONNX export |
| **Developer** | Full REST API, Docker, open source — plug into anything |

---

## 🃏 CARD 10 — Open Source

- **GitHub:** `github.com/ottco-dev/ctip-oss`
- MIT license (planned)
- Runs fully **offline / air-gapped** — no data leaves your machine
- Local VLMs (Moondream, Florence-2) — no API key required to start
- Docker Compose → one command to run everything

---

## 🃏 CARD 11 — Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU mode) | RTX 4060 8 GB |
| RAM | 8 GB | 16 GB |
| CPU | Any modern | i5-13400F or better |
| Microscope | Any USB camera | 40–100× with camera port |
| OS | Linux / macOS / Windows | Ubuntu 22.04 |

**Cost to run:** A used RTX 4060 + a USB microscope adapter (~$50) is all the hardware needed beyond a desktop PC.

---

## 🃏 CARD 12 — Numbers to Remember

| Metric | Value |
|---|---|
| Automated tests | **1,650** |
| API endpoints | **80+** |
| Frontend pages | **20+** |
| Supported VLM providers | **9** (local + cloud) |
| Wiki pages | **20** (EN / DE / ES) |
| Inference speedup (batch) | **~3×** vs single-image |
| Languages (UI + wiki) | **English / German / Spanish** |

---

## 🃏 CARD 13 — Likely Questions & Answers

**Q: Does it work with any microscope?**
> Any USB camera or video source works. Better optics = better results.
> We recommend 40–100× magnification with good lighting.

**Q: Can it predict THC percentage?**
> No. Optical trichome maturity is NOT a proxy for cannabinoid concentration.
> We're explicit about this in the code and UI. We describe what we see.

**Q: Is the model pre-trained or do I need my own data?**
> We provide a YOLO v11s architecture ready for fine-tuning.
> The annotation + training pipeline is built-in — Label Studio, CVAT, active learning.

**Q: Does it run in the cloud?**
> Designed for local/on-premise. Can be deployed anywhere with Docker.
> No mandatory cloud dependencies.

**Q: What's the accuracy?**
> We don't quote accuracy without your specific dataset.
> The framework is built to be fine-tuned on your microscope, your strain, your lighting.

**Q: Is it free?**
> Open source. Runs on hardware you probably already own.

---

## 🔑 TOP 3 TAKEAWAYS (close with these)

1. **Consistent** — same image → same result, every time, for every grower
2. **Transparent** — open source, local, no black box, human always in control
3. **Scientific** — built like research infrastructure, not a phone app

---

*CTIP · Cannabis Trichome Intelligence Platform · ottco-dev/ctip-oss*
