# Trichome Microscopy — Image Acquisition Methodology

**Version**: 1.0  
**Date**: 2026-05-26  
**Scope**: Brightfield and reflected-light stereo microscopy for Cannabis trichome imaging  
**Related**: [calibration_protocols.md](calibration_protocols.md) · [../evaluation_methodology/trichome_evaluation_methodology.md](../evaluation_methodology/trichome_evaluation_methodology.md)

---

## 1. Purpose and Scope

This document defines the image acquisition methodology for training data generation and
production inference in the Trichome Analysis Platform. Consistent acquisition is required
because:

- Detection model performance degrades with out-of-distribution illumination
- Maturity classification depends on colour channels that are illumination-sensitive
- Calibrated measurement (µm) requires stable, documented magnification
- Scientific reproducibility demands the same protocol across sessions and operators

The protocol is designed for low-cost benchtop microscopes with USB/CSI cameras. It does
NOT require research-grade confocal or fluorescence equipment.

**Scientific constraint**: No protocol described here, and no optical observation derived
from it, permits inference about cannabinoid concentration. Maturity stage is an optical
observation only.

---

## 2. Hardware Requirements and Recommendations

### 2.1 Microscope Types

| Type | Objective Range | Trichome Use Case | Notes |
|---|---|---|---|
| Stereo dissecting microscope | 10×–80× | Whole-plant surveys | Wide FOV; low working distance |
| Compound brightfield | 4×–100× | High-magnification head detail | Standard; use cover glass |
| Digital USB microscope | 20×–200× | Field/harvest screening | Inconsistent illumination; calibrate each session |
| Reflected-light (metallurgical) | 5×–50× | Opaque specimens without clearing | Eliminates coverslip artifacts |

**Recommended minimum**:
- 10× and 40× objectives
- 1:1 parfocal system (same focus plane at both objectives)
- Mechanical stage with X/Y micrometer for reproducible positioning

### 2.2 Camera Requirements

| Parameter | Minimum | Recommended |
|---|---|---|
| Resolution | 1920 × 1080 | 3840 × 2160 (4K) |
| Sensor format | 1/2.9" | 1/2" or larger |
| Bit depth | 8-bit | 12-bit RAW |
| Frame rate (for video) | 30 fps | 60 fps |
| Interface | USB 2.0 | USB 3.0 / CSI-2 |
| Colour filter | RGB Bayer | RGB Bayer |
| Example sensors | OV5647 | Sony IMX477 (Raspberry Pi HQ) |

**Note**: 8-bit RGB is adequate for training data. 12-bit RAW is preferred for
quantitative colour measurement (maturity staging) because it preserves the dynamic range
of the cloudy–amber colour transition.

### 2.3 Illumination Sources

| Source | Spectral Quality | Flicker | CRI | Recommendation |
|---|---|---|---|---|
| Tungsten halogen (transmitted) | Warm (3200K) | None | ~100 | ✅ Standard for transmitted light |
| Cold white LED ring | Cool (5500–6500K) | Potential 50/60 Hz | 80–90 | ✅ With flicker check (see §5.2) |
| Warm white LED ring | Warm (3000K) | Potential | 80–90 | ⚠️ May oversaturate amber trichomes |
| Fibre-optic dual gooseneck | Adjustable | None | ~95 | ✅ Best for oblique/reflected |
| Fluorescent ring | Mixed | 100/120 Hz | 70–80 | ❌ Avoid — flicker + poor CRI |
| UV illumination | Narrow band | None | N/A | ❌ Not used (no fluorescence protocol) |

---

## 3. Specimen Preparation

### 3.1 Plant Material Types

| Material | Preparation | Notes |
|---|---|---|
| Living tissue (in-vivo) | None | Highest quality; must image quickly |
| Fresh cut bract/calyx | Place on microscope slide | Viable 30–60 min before wilting |
| Dried & cured material | Rehydrate 5 min in humid chamber | Reduces shrinkage artifacts |
| Extracted resin | Press between slides | High trichome density; useful for counting |
| Preserved (70% EtOH) | Remove EtOH; air-dry briefly | Acceptable colour preservation |

### 3.2 Slide Preparation

1. Lay the specimen flat on a clean glass slide
2. If using a cover glass: apply 1 drop of distilled water, lower cover glass at an angle
   to avoid air bubbles
3. For reflected-light imaging: no cover glass required; place specimen directly
4. Label the slide with: `plant_id`, `session_id`, `date`, `objective`

### 3.3 Avoiding Common Preparation Artefacts

| Artefact | Cause | Prevention |
|---|---|---|
| Air bubbles under coverslip | Rapid mounting | Lower coverslip at angle; use aqueous mounting medium |
| Desiccation halos | Dried specimen | Rehydrate or image fresh |
| Refractive rings around stalks | Dry air gap | Use water as mounting medium |
| Specimen movement during acquisition | Insufficient compression | Gently press coverslip; use nail varnish sealant for archival |
| Chromatic aberration at stalk edges | Objective quality | Use plan-apochromatic objectives; avoid economy objectives |

---

## 4. Illumination Protocol

### 4.1 Köhler Illumination (Transmitted Light)

Köhler illumination eliminates the filament image from the specimen plane and provides
even, controlled illumination. It is required for quantitative colour work.

**Setup procedure**:
1. Focus on the specimen at the working objective
2. Fully open the field diaphragm
3. Close the condenser aperture diaphragm to ~2/3 open
4. Rack the condenser up until you see the field diaphragm image in focus in the specimen plane
5. Centre the condenser using the centring screws
6. Open the field diaphragm until it just disappears outside the field of view
7. Adjust the aperture diaphragm for the desired contrast (NA matching)
8. **Do NOT change illumination intensity after this step** — adjust camera exposure instead

### 4.2 Reflected Light (Epi-illumination)

For stereo and reflected-light microscopes:
1. Position gooseneck or ring light at 45° to the specimen plane
2. Bilateral illumination (two sources, one per side) reduces hard shadows on stalked trichomes
3. Diffuse the light source with a sheet of translucent white PTFE tape if shadows are harsh
4. Check: no specular reflections on trichome heads (if present, lower light angle or add polariser)

### 4.3 Illumination Stability Requirements

| Parameter | Requirement | Test |
|---|---|---|
| Warm-up time | ≥ 15 min after power-on | Record mean intensity first vs. 15-min pixel |
| Intensity drift within session | < 2% | Capture blank slide at session start and end; compare mean pixel |
| Flicker (LED sources) | < 1% amplitude at capture FPS | Capture blank slide at 100 fps; compute temporal std |
| Spatial uniformity | ± 10% corner-to-centre | Blank slide flat-field image; histogram per quadrant |

### 4.4 Flat-Field Correction

For quantitative colour work, apply flat-field correction before analysis:

```
I_corrected = (I_raw - I_dark) / (I_flat - I_dark)
```

Where:
- `I_raw`: raw specimen image
- `I_dark`: camera noise floor (lens cap, same exposure)
- `I_flat`: blank-slide image under identical illumination

Implementation: `analytics/preprocessing/flat_field.py` → `apply_flat_field_correction()`

**Threshold**: Apply flat-field correction when corner-to-centre intensity difference > 5%.

---

## 5. Objective Selection and Magnification

### 5.1 Objective Selection per Use Case

| Objective | FOV (Sony IMX477, 4K) | µm/px | Primary Use |
|---|---|---|---|
| 4× | ~2.4 × 1.8 mm | 0.39 | Whole-bract survey; density estimation |
| 10× | ~0.95 × 0.72 mm | 0.16 | Standard trichome counting field |
| 20× | ~0.48 × 0.36 mm | 0.078 | Head morphology; type classification |
| 40× | ~0.24 × 0.18 mm | 0.039 | Maturity staging; detailed colour |
| 100× (oil) | ~0.10 × 0.07 mm | 0.016 | Research only; requires oil immersion |

Values assume Sony IMX477 sensor (1.55 µm pixel pitch). Recalibrate for other sensors.

### 5.2 Matching Objective to Detection Model

The YOLO detection model was trained on a mixture of magnifications. The following pixel
size ranges are acceptable for production inference:

| Trichome Size Class | Minimum µm/px | Maximum µm/px | Recommended Objective |
|---|---|---|---|
| Bulbous glandular (30–80 µm) | 0.08 | 0.40 | 10× or 20× |
| Sessile glandular (60–120 µm) | 0.06 | 0.40 | 10× or 20× |
| Stalked glandular (120–500 µm) | 0.04 | 0.50 | 4× or 10× |
| Non-glandular (various) | 0.06 | 0.50 | 4× or 10× |

**Hard constraint**: Do NOT use 100× for training data unless bulbous trichomes are the
sole target. At 100×, stalked trichomes are larger than the detector FOV.

### 5.3 Numerical Aperture and Resolution

Rayleigh resolution limit:
```
d = 0.61 × λ / NA
```
At NA=0.25 (10× objective), λ=550 nm: d ≈ 1.34 µm — resolves the neck of stalked trichomes.
At NA=0.10 (4× objective): d ≈ 3.4 µm — resolves head boundary but not fine stalk detail.

The detection model does not require sub-diffraction resolution; head boundary detection at
4× is sufficient for all classification tasks.

---

## 6. Focus and Depth of Field

### 6.1 Depth of Field Values

| Objective (NA) | Total depth of field |
|---|---|
| 4× (NA 0.10) | ~55 µm |
| 10× (NA 0.25) | ~8.8 µm |
| 20× (NA 0.40) | ~3.4 µm |
| 40× (NA 0.65) | ~1.3 µm |

Trichome stalks are 50–300 µm tall; the entire structure cannot be in focus simultaneously
at ≥ 20×. For training data, focus on the **trichome head** (the secretory gland, not the
base or stalk).

### 6.2 Focus Protocol

For static single-image acquisition:
1. Bring the focal plane to the specimen surface at low magnification (4×)
2. Increase to target objective
3. Adjust focus until the majority of trichome heads in the field are sharp
4. Accept images where > 50% of annotatable trichomes are in focus
5. Reject images where the best-focus plane is on the stalk base or mounting glass

**Reject threshold for training data**: fewer than 3 in-focus annotatable trichomes in the frame.

### 6.3 Extended Depth of Field (EDOF)

EDOF by z-stacking is NOT used in production inference because:
1. It requires mechanised Z-drive
2. Processing latency is incompatible with real-time inference
3. The detector is robust to moderate focus variation within the training distribution

EDOF is optionally used for morphology benchmark images. Protocol:
- Z-step: 2 µm at 10×, 1 µm at 40×
- Stack depth: ±20 µm from head centre
- Fusion algorithm: focus-stacking by variance (Laplacian)

---

## 7. Exposure and Camera Settings

### 7.1 Exposure Guidelines

| Parameter | Value | Rationale |
|---|---|---|
| Target mean pixel (8-bit) | 120–160 (of 255) | Avoids clipping; maintains headroom |
| Maximum allowed saturation | < 1% of pixels at 255 | Saturated trichome heads lose colour info |
| Minimum mean pixel | > 80 | Avoids noise-dominated images |
| Auto-exposure | OFF | Must not change between frames in a session |
| Auto white balance | OFF | Colour-critical for maturity staging |
| Gain / ISO | Fix at minimum | Amplifies noise; set exposure time instead |

### 7.2 White Balance

White balance must be set **once per session** using a reference target:
1. Place a piece of spectralon or matte white paper under the objective
2. Use the "auto white balance" function once only (to derive R/G/B gains)
3. Lock the gains — do NOT allow adaptive AWB during data collection
4. Record the R/G/B gains in the session metadata

For maturity staging (cloudy vs. amber), white balance consistency is critical.
A ΔE shift > 3 CIE Lab units between sessions will degrade classification accuracy.

### 7.3 File Format

| Use Case | Format | Bit Depth | Notes |
|---|---|---|---|
| Training data (labels) | PNG (lossless) | 8-bit RGB | JPEG compression artifacts corrupt small features |
| Quantitative colour analysis | TIFF (lossless) | 16-bit | Preserves full camera bit depth |
| Archive / reference | TIFF | 16-bit | Required for measurement traceability |
| Inference (production) | JPEG acceptable | 8-bit | Model trained on 8-bit; acceptable for detection |

**Do NOT use JPEG for training data or calibration images.**

---

## 8. Session Metadata Requirements

Every acquisition session MUST record the following metadata. These fields are required
for leakage-safe dataset splitting and calibration traceability.

### 8.1 Mandatory Fields

| Field | Type | Example |
|---|---|---|
| `session_id` | UUID | `sess-2026-05-26-001` |
| `plant_id` | String | `cultivar-A-plant-003` |
| `acquisition_date` | ISO 8601 | `2026-05-26T14:32:00Z` |
| `operator_id` | String | `op-01` |
| `microscope_model` | String | `Amscope B120C` |
| `objective_magnification` | Integer | `40` |
| `objective_na` | Float | `0.65` |
| `camera_model` | String | `Raspberry Pi HQ v2` |
| `sensor_pixel_size_um` | Float | `1.55` |
| `um_per_pixel` | Float (measured) | `0.041` |
| `illumination_type` | Enum | `transmitted_led` |
| `illumination_kelvin` | Integer | `5500` |
| `exposure_ms` | Float | `12.4` |
| `white_balance_r` | Float | `1.42` |
| `white_balance_b` | Float | `1.87` |
| `flat_field_applied` | Boolean | `true` |
| `calibration_reference` | String | `stage-micrometer-10um-2026-05-26` |

### 8.2 Optional Fields (Recommended)

| Field | Type | Description |
|---|---|---|
| `specimen_preparation` | Enum | `fresh` \| `dried` \| `rehydrated` \| `preserved` |
| `cover_glass_used` | Boolean | Affects µm/px slightly via mounting medium |
| `mounting_medium` | String | `water` \| `glycerol` \| `none` |
| `plant_phenological_stage` | Enum | `vegetative` \| `early_flower` \| `mid_flower` \| `late_flower` |
| `z_position_um` | Float | Z-stage position if using motorised stage |
| `environmental_temperature_c` | Float | Affects tissue hydration |
| `notes` | Text | Free-form; artefacts, anomalies |

---

## 9. Calibration Per Session

### 9.1 Required Calibration Procedure

Before each session (or whenever objective or camera is changed):

1. Prepare a stage micrometer with known pitch (10 µm divisions recommended)
2. Place the micrometer under the same objective and illumination as the specimen
3. Acquire 3 images of the scale bar field
4. Measure the pixel span of a known number of divisions (minimum 10 divisions)
5. Compute µm/px: `µm_per_px = (N × pitch_µm) / pixel_span`
6. Average across the 3 images; compute SD
7. Reject if SD / mean > 0.5% (suggests stage drift or vibration)
8. Log calibration to `calibration_artifacts/session_id/` and to the session metadata

See [calibration_protocols.md](calibration_protocols.md) for the full calibration procedure and uncertainty quantification.

### 9.2 Calibration Stability Checks

| Check | Frequency | Threshold |
|---|---|---|
| µm/px repeatability | Per objective change | SD/mean < 0.5% |
| White balance drift | Per session | ΔE < 3 CIE Lab units |
| Illumination intensity drift | Start vs. end of session | < 2% mean pixel change |
| Stage micrometer accuracy | Quarterly | Compare to NIST-traceable reference |

---

## 10. Image Quality Gates

### 10.1 Automated Quality Filter

Images are processed through `shared/preprocessing/quality_filter.py` before entering
the training or inference pipeline. The following thresholds gate acceptance:

| Metric | Reject Threshold | Implementation |
|---|---|---|
| Mean pixel brightness | < 60 or > 210 (8-bit) | Global mean |
| Saturation fraction | > 3% of pixels at 255 | Per-channel max |
| Blur score (Laplacian variance) | < 50 | `cv2.Laplacian(gray, ddepth=cv2.CV_64F).var()` |
| Noise (BRISQUE-like) | > 70 (0–100 scale) | BRISQUE or equivalent |
| Tissue coverage | < 10% of frame | Otsu threshold + fraction |

### 10.2 Manual Quality Annotation

For training data, annotators must tag each image with:

| Tag | Meaning |
|---|---|
| `quality:excellent` | Köhler illumination, sharp focus, good exposure |
| `quality:good` | Minor focus variation; full annotation possible |
| `quality:marginal` | Used only for hard-negative mining, not standard training |
| `quality:reject` | Artefact / out-of-distribution; discard |

Only `quality:excellent` and `quality:good` images enter the training set.

---

## 11. Acquisition Workflow

### 11.1 Session Start Checklist

- [ ] Microscope warmed up ≥ 15 min
- [ ] Objective selected and recorded in metadata
- [ ] Köhler illumination set (transmitted) or gooseneck positioned (reflected)
- [ ] Camera exposure fixed (auto-exposure OFF)
- [ ] White balance set and locked (auto-WB OFF)
- [ ] Dark frame captured (lens cap, same exposure)
- [ ] Flat-field frame captured (blank slide)
- [ ] Stage micrometer calibration completed and logged
- [ ] Session metadata file created with `session_id`, `plant_id`, `operator_id`

### 11.2 Per-Field Acquisition Protocol

1. Position field of view; ensure ≥ 3 annotatable trichomes are visible
2. Confirm focus is on trichome heads, not stalk bases or slide surface
3. Wait 1 s for vibration to settle after mechanical adjustment
4. Capture single frame (or 3-frame median stack for noise reduction)
5. Verify image quality visually (blur, exposure, coverage)
6. Record `z_position_um` if motorised stage is available
7. Move to next field; minimum 20% overlap between adjacent fields for tiled coverage

### 11.3 Session End

- [ ] Capture second flat-field frame (session drift check)
- [ ] Capture second stage micrometer calibration image (verify stability)
- [ ] Export all images + metadata to `data/raw/sessions/<session_id>/`
- [ ] Run `scripts/ingest_session.py --session-id <session_id>` to trigger:
  - Quality filtering
  - pHash deduplication
  - Metadata validation
  - Addition to the annotation queue

---

## 12. Special Considerations for Maturity Imaging

Trichome maturity staging (clear → cloudy → amber) is colour-dependent. The following
additional constraints apply when images are acquired specifically for maturity work:

### 12.1 Colour Accuracy Requirements

- White balance reference: **D65 standard illuminant** (6500K equivalent) or as close as available
- CRI (Colour Rendering Index) of illumination source: ≥ 85
- Do NOT use warm-white illumination (< 4000K) for maturity imaging — the amber classification signal is severely degraded
- Apply flat-field correction unconditionally for maturity training data

### 12.2 Amber vs. Cloudy Boundary

The amber–cloudy transition is the hardest classification boundary. Optimal conditions:

- 40× objective (0.039 µm/px at Sony IMX477)
- Transmitted illumination (head translucency is the key signal)
- Slightly reduced condenser aperture (NA ~0.5× objective NA) for higher contrast
- Avoid oblique or reflected-only illumination — it creates pseudocolour from interference

### 12.3 Mandatory Scientific Caveat

All maturity reports, annotations, and model outputs **must** include:

> *"Maturity stage is an observable optical property of trichome head colour and
> translucency. No inference about cannabinoid concentration can be made from
> visual appearance alone."*

This caveat is enforced at the API response level in `maturity/api/router.py`.

---

## 13. Training Data Diversity Requirements

To ensure the detection model generalises across acquisition conditions, the training
dataset must include images from diverse conditions:

| Variation | Minimum Coverage | Target Coverage |
|---|---|---|
| Objectives (4×, 10×, 20×, 40×) | All 4 present | Balanced ±20% |
| Illumination types (transmitted, reflected) | Both present | 60% transmitted, 40% reflected |
| Specimen preparation (fresh, dried, preserved) | Fresh + dried | All 3 types |
| White balance colour temperatures (4000–6500K) | 2 temperatures | ≥ 3 temperatures |
| Operator IDs | ≥ 2 operators | ≥ 3 operators |
| Plant IDs | ≥ 10 plants | ≥ 30 plants |
| Sessions | ≥ 10 sessions | ≥ 50 sessions |

Deficiencies in any dimension increase the risk of systematic model bias.

---

## 14. Relationship to Dataset Pipeline

```
Acquisition session
    → data/raw/sessions/<session_id>/
    → scripts/ingest_session.py
         → quality filter (§10.1)
         → pHash dedup
         → metadata validation (§8.1)
         → annotation queue (Label Studio)
    → human annotation (HITL mandatory)
    → data/annotated/<session_id>/
    → dataset builder → train/val/test split (by plant_id, session_id)
    → training pipeline
```

Dataset split MUST be by `plant_id` and `session_id`, never by image ID.
See [../evaluation_methodology/trichome_evaluation_methodology.md §6](../evaluation_methodology/trichome_evaluation_methodology.md) for the leakage prevention protocol.

---

## 15. References

1. Murphy, D.B. (2001). *Fundamentals of Light Microscopy and Electronic Imaging*. Wiley-Liss.
2. ISO 9345-1 (2012). *Micrographic imaging — Image quality assessment*. ISO.
3. BIPM (2008). *Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM)*. JCGM 100:2008.
4. Russ, J.C. (2011). *The Image Processing Handbook*, 6th ed. CRC Press.
5. Abramowitz, M. & Davidson, M.W. *Köhler Illumination*. Olympus Microscopy Resource Center. [Online resource]
