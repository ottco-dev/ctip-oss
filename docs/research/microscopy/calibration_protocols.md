# Microscopy Calibration Protocols

## Purpose

Calibration is required to convert pixel measurements to physical units (µm).
Without calibration, all morphological measurements are in pixel units only
and are **not scientifically comparable** across different microscopes,
objectives, or camera sensors.

This document describes:
1. The theoretical basis for pixel-to-µm calibration
2. Three calibration methods (stage micrometer, known object, theoretical)
3. Step-by-step protocols for each method
4. Uncertainty sources and how to minimize them
5. Calibration validation procedure

---

## 1. Theoretical Background

### 1.1 Image Scale and Magnification

The pixel size in µm (µm/px) depends on:

```
µm_per_pixel = sensor_pixel_size_µm / (objective_magnification × tube_lens_factor)
```

Where:
- `sensor_pixel_size_µm`: Physical size of each pixel on the camera sensor (from sensor spec sheet)
- `objective_magnification`: Objective power (4×, 10×, 20×, 40×, 100×)
- `tube_lens_factor`: Correction factor for tube lens (1.0 for standard 160mm tube; varies for infinity-corrected systems)

### 1.2 Common Values

| Objective | Sensor px (Sony IMX477) | µm/px |
|-----------|------------------------|-------|
| 4×        | 1.55 µm                | 0.388 |
| 10×       | 1.55 µm                | 0.155 |
| 20×       | 1.55 µm                | 0.078 |
| 40×       | 1.55 µm                | 0.039 |
| 100×      | 1.55 µm                | 0.016 |

| Objective | Sensor px (standard 2.4µm) | µm/px |
|-----------|---------------------------|-------|
| 4×        | 2.4 µm                    | 0.600 |
| 10×       | 2.4 µm                    | 0.240 |
| 20×       | 2.4 µm                    | 0.120 |
| 40×       | 2.4 µm                    | 0.060 |
| 100×      | 2.4 µm                    | 0.024 |

**Note**: Theoretical values assume no tube lens magnification and no camera adapter magnification.
Always validate theoretical values against a physical reference.

---

## 2. Method 1: Stage Micrometer (Recommended — Highest Accuracy)

A stage micrometer is a microscope slide with a precision-etched scale bar.
Standard: 1 mm ruled in 100 divisions = 10 µm per division.
NIST-traceable stage micrometers: ±0.5 µm absolute accuracy.

### 2.1 Equipment

- Stage micrometer (NIST-traceable preferred)
  - Example: Edscientific TS-102, 1mm/100div
  - Example: Amscope MR095, 1mm/0.01mm
- Same microscope + objective + camera as intended imaging sessions
- Same acquisition software and settings (bit depth, binning, white balance)

### 2.2 Protocol

**Step 1: Setup**
1. Warm up microscope lamp for ≥ 15 minutes (thermal stability)
2. Set Köhler illumination
3. Select target objective (calibrate per objective)
4. Set camera gain, exposure, and white balance identically to sample conditions

**Step 2: Image acquisition**
1. Place stage micrometer on stage
2. Focus sharply on the graduation marks
3. Capture 5–10 images at different positions along the scale
4. Record: objective, tube length, camera settings, date, temperature

**Step 3: Measurement**
```python
# Using the trichome platform CLI:
trichome calibrate run \
  --image micrometer_40x_001.tif \
  --reference-length-um 100.0 \
  --auto  # Uses Hough line detection

# Or interactive mode:
trichome calibrate run \
  --image micrometer_40x_001.tif \
  --reference-length-um 100.0
# Click start and end of the 100µm scale bar when prompted
```

**Step 4: Average across images**
```python
measurements_um_per_px = [0.1623, 0.1626, 0.1624, 0.1625, 0.1622]
um_per_pixel = statistics.mean(measurements_um_per_px)    # 0.1624
uncertainty = statistics.stdev(measurements_um_per_px)    # 0.0002
```

**Step 5: Save profile**
```python
# The CLI auto-saves calibrated profiles:
trichome calibrate list  # View saved profiles
trichome calibrate show 40x_olympus_bx53
```

### 2.3 Uncertainty Budget (Stage Micrometer)

| Source | Magnitude | Type |
|--------|-----------|------|
| NIST-traceable micrometer accuracy | ±0.5 µm / 100 µm | B |
| Pixel edge detection | ±0.5 px × µm/px | A |
| Thermal drift (lamp heating) | ~0.01% per °C | B |
| Focus accuracy | ~1 px lateral error | A |
| Measurement repetition (N=5) | σ/√5 | A |

**Combined expanded uncertainty (k=2, 95%)**: typically ±1–3% for 40× objective.

---

## 3. Method 2: Known Reference Object

Use a biological structure or standard bead of known size.

### 3.1 Suitable references

| Reference | Size | Notes |
|-----------|------|-------|
| Fluorescent calibration beads (2.0µm) | 2.0 µm diameter | Expensive but very accurate |
| Pollen grains (ragweed: Ambrosia) | ~20 µm | Widely available, check literature size |
| Styrene microspheres | 5–100 µm (catalog) | Sigma-Aldrich spheres certified ±2% |
| Etched glass grid | 10–100 µm grids | Commercial; similar to stage micrometer |

### 3.2 Protocol

Same as stage micrometer but measure the known reference feature instead of a ruled line.

**Accuracy**: Lower than NIST-traceable stage micrometer.
Typical uncertainty: ±5–10% depending on reference material quality.

---

## 4. Method 3: Theoretical (Least Accurate)

Use manufacturer specifications to calculate µm/px without physical calibration.

```bash
trichome calibrate estimate \
  --objective 40 \
  --pixel-size-um 2.4 \
  --tube-lens 1.0
```

**When to use**: Quick approximation only. Never for published measurements.
**Accuracy**: ±10–20% due to:
- Actual sensor pixel size may differ ±5%
- Tube lens factor is often undocumented or non-standard
- Camera adapter magnification not accounted for
- Objective magnification tolerance ±2% (manufacturer spec)

---

## 5. Multi-Objective Calibration Strategy

For systems using multiple objectives:

```bash
# Calibrate each objective independently
trichome calibrate run --image micro_10x.tif --profile-name "10x_setup_A" --reference-length-um 100
trichome calibrate run --image micro_20x.tif --profile-name "20x_setup_A" --reference-length-um 50
trichome calibrate run --image micro_40x.tif --profile-name "40x_setup_A" --reference-length-um 25

# Check linearity: µm/px should scale inversely with magnification
trichome calibrate list
```

**Expected linearity test**: `(µm/px at 10×) / (µm/px at 40×)` ≈ 4.0 ± 0.05
Deviation > 5% indicates a problem (wrong objective labeled, camera adapter, etc.)

---

## 6. Calibration Validation

After calibrating, validate using an independent reference:

```bash
trichome calibrate run \
  --image validation_sample.tif \
  --profile "40x_setup_A"
```

**Cross-check**: Measure 3–5 trichomes of known morphological type and compare
to published size ranges:
- Bulbous: 10–30 µm head diameter
- Capitate-sessile: 25–100 µm head diameter  
- Capitate-stalked: 60–120 µm head diameter, 100–500 µm total height

If measured values fall ≥ 30% outside published ranges:
1. Recheck objective identification
2. Recheck camera sensor specification
3. Repeat stage micrometer calibration

---

## 7. Session Consistency

For longitudinal studies:
- Calibrate at the **start of each imaging session**
- Record: date, microscope ID, objective, lamp age (hours), ambient temperature
- Use the same illumination setting; lamp aging affects Köhler alignment
- Store calibration profiles per session in `~/.trichome/profiles/`

---

## 8. Implementation Reference

| Component | File |
|-----------|------|
| Scale bar auto-detection | `measurement/calibration/stage_micrometer.py` |
| Profile management (CRUD) | `measurement/domain/profile_manager.py` |
| px → µm conversion | `measurement/domain/measurer.py` |
| GUM uncertainty propagation | `measurement/domain/propagation.py` |
| Measurement pipeline | `measurement/application/measurement_pipeline.py` |
| Calibration CLI | `apps/cli/commands/calibrate.py` |

---

## 9. References

1. **Abramowitz, M. & Davidson, M.W.** (2012). Microscopy Resource Center: Calibration and Measurement. Olympus America Inc.
   https://www.microscopyu.com/microscopy-basics/resolution

2. **JCGM 100:2008**. Evaluation of measurement data — Guide to the Expression of Uncertainty in Measurement (GUM). BIPM, Sèvres, France.
   https://www.bipm.org/utils/common/documents/jcgm/JCGM_100_2008_E.pdf

3. **ISO 10110-7:2017**. Optics and photonics — Preparation of drawings for optical elements and systems — Part 7: Error of form and position.

4. **Murphy, D.B. & Davidson, M.W.** (2012). Fundamentals of Light Microscopy and Electronic Imaging, 2nd ed. Wiley-Blackwell.
   ISBN: 978-0-471-69214-0

5. **Inoué, S. & Spring, K.R.** (1997). Video Microscopy: The Fundamentals, 2nd ed. Plenum Press, New York.
   ISBN: 978-0-306-45531-0
