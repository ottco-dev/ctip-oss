# Trichome Morphology Science

## 1. Botanical Classification

Cannabis sativa L. produces three morphologically distinct glandular trichome types
plus one non-glandular type. Understanding the biology of each type is essential
for building accurate classifiers.

### 1.1 Capitate-Stalked Trichomes (Capitate Glandular)

**Morphology:**
- Multicellular structure: secretory disc cells + stalk cells
- Stalk: 100–500 µm tall, 2–5 cells thick
- Head (capitulum): 50–100 µm diameter, spherical to oval
- Head comprises 8–16 secretory disc cells + single-cell cover
- Attached to substrate by a single basal cell

**Biochemistry:**
- Primary site of cannabinoid biosynthesis (THCA, CBDA, terpenes)
- Stalk contains no cannabinoids — secretory activity localized to head
- Head cavity (sub-cuticular space) accumulates exudate during maturation

**Visual characteristics:**
- Clear → milky → amber progression during maturation
- Head color/opacity primary maturity indicator
- Stalk visible under 10-40× magnification

**Distribution:**
- Highest density on pistillate floral bracts and subtending leaves
- Lower density on fan leaves, stems
- ~60% of observed trichomes in typical close-up images

**Measurement targets:**
- Head diameter: 50-100 µm (validation target for calibration)
- Stalk length: variable (100-500 µm)
- Head:stalk aspect ratio: useful for maturity assessment

---

### 1.2 Capitate-Sessile Trichomes

**Morphology:**
- Similar to capitate-stalked but stalk is very short or absent
- Head: 25–75 µm diameter
- Sits nearly flush with epidermal surface
- Smaller than capitate-stalked

**Biochemistry:**
- Active in cannabinoid synthesis, but secondary to capitate-stalked
- Lower total cannabinoid content per trichome than capitate-stalked

**Visual characteristics:**
- Appears as rounded protrusion without visible stalk
- Often confused with late-stage capitate-stalked (after stalk degradation)
- Maturity progression similar to capitate-stalked

**Distribution:**
- Common on upper leaf surfaces, bracts
- ~25% of observed trichomes

**Classification challenge:**
- Capitate-sessile vs. capitate-stalked overlap at low magnification (10×)
- Reliable separation requires 40×+ objective
- At 10×: sessile appears as "dot", stalked as "lollipop"

---

### 1.3 Bulbous Trichomes

**Morphology:**
- Smallest glandular type: 10–30 µm diameter
- Consists of 2–4 cells: 1 basal + 1–3 apical
- No visible stalk at standard microscopy magnification
- Spherical to slightly elongated

**Biochemistry:**
- Limited cannabinoid content (debate in literature)
- May function primarily as mechanical deterrent

**Visual characteristics:**
- Near-transparent to slightly opaque
- Very small — requires 40×+ for reliable detection
- At 20×: appears as tiny circular dot
- Easily confused with leaf surface features

**Distribution:**
- Present on all vegetative surfaces
- ~10% of observed trichomes in typical floral images
- **Rare class challenge**: requires focal loss / class reweighting

**Detection difficulty:**
- Size: 10-30 µm → 5-20 pixels at 10× magnification
- Below COCO "small object" threshold (32×32 px)
- Requires 1280px input and tiled inference for reliable detection

---

### 1.4 Non-Glandular (Cystolithic / Unicellular) Trichomes

**Morphology:**
- Single elongated cell: 100-400 µm long
- Hair-like, pointed tip
- No secretory apparatus
- Contains calcium oxalate crystals (cystoliths) in some types

**Biochemistry:**
- No cannabinoid production
- Mechanical function: herbivore deterrence, UV protection

**Visual characteristics:**
- Elongated, hair-like
- No head/capitulum
- Color: transparent to white
- Easy to distinguish from glandular types at 10×

**Distribution:**
- Common on leaf margins, stem surfaces
- ~5% of observed trichomes in floral images

**Classification note:**
- Rarest class — requires aggressive class reweighting (6× in weighted sampler)
- True negatives for maturity analysis (exclude from maturity scoring)

---

## 2. Maturation Biology

### 2.1 Developmental Stages

Trichome maturation follows a consistent optical progression:

| Stage | Head Appearance | Biological State |
|-------|----------------|-----------------|
| Clear | Transparent, glass-like | Resin accumulation phase; THCA being synthesized |
| Cloudy | Milky white, opaque | Peak resin density; Mie scattering of resin droplets |
| Amber | Orange-brown tint | Oxidative degradation; THC → CBN conversion |
| Mixed | Heterogeneous | Population with multiple stages present |

### 2.2 Mie Scattering — Why Trichomes Turn Cloudy

The transition from clear to cloudy is explained by optical physics, not chemistry:

1. As resin density increases, droplet size and concentration cross a threshold where
   **Mie scattering** dominates over Rayleigh scattering
2. Mie scattering: occurs when scatterer diameter ≈ photon wavelength
3. Dense resin droplets scatter all visible wavelengths equally → white/opaque appearance
4. Additionally, THCA crystallization increases refractive index contrast → more scattering

**Critical implication**: Cloudy appearance measures OPTICAL DENSITY, not chemical concentration.
Two trichomes of identical cloudiness may contain dramatically different THCA amounts.

Reference: Fischedick et al. (2010). Phytochemistry 71(17-18):2058-2073.

### 2.3 Amber Coloration — The Degradation Mechanism

Amber coloration results from photo-oxidative degradation:

**Mechanism** (ElSohly & Slade, 2005):
1. UV exposure catalyzes dehydrogenation of the pyran ring in THC
2. Step 1: THC → Δ8-THC (isomerization)
3. Step 2: Δ8-THC → CBN (complete aromatization of the ring)
4. CBN has an orange-brown chromophore → amber coloration

**Practical consequences**:
- Amber % = oxidation signal, NOT THC% indicator
- High amber % means cannabinoids are degrading
- Cultivators use amber% as "harvest now or lose potency" signal
- This is valid as a QUALITATIVE indicator only

---

## 3. Microscopy Requirements

### 3.1 Magnification Recommendations

| Objective | FOV | Pixel/µm at 4K | Visible details |
|-----------|-----|----------------|-----------------|
| 4× | 3.5×2.6 mm | 0.65 | Overall bud structure |
| 10× | 1.4×1.1 mm | 0.26 | All trichome types visible |
| 20× | 0.7×0.5 mm | 0.13 | Head/stalk detail |
| 40× | 350×260 µm | 0.065 | Cell-level detail, bulbous |

**Recommended**: 10-20× for detection; 40× for measurement

### 3.2 Focus Requirements

Trichomes have narrow depth of field. Focus quality directly affects detection:
- Laplacian variance target: > 80 for reliable detection
- Focus stacking recommended for 3D characterization
- Single-plane focus: analyze only in-focus region

### 3.3 Illumination

- Backlit (transmitted): shows head transparency clearly
- Reflected: shows surface texture, color
- **Best for maturity**: reflected light with slight darkfield component
- Avoid UV illumination during analysis (causes real-time photooxidation)

---

## 4. Morphological Measurements

### 4.1 Head Diameter

Primary quantitative measurement for trichome characterization.

- Measurement method: maximum caliper diameter of the capitulum
- Reference: 50-100 µm for capitate-stalked (calibrate against stage micrometer)
- Circularity metric useful: circularity = 4π × area / perimeter²
  - Bulbous: circularity ≈ 0.85-0.95 (near-spherical)
  - Capitate-stalked: circularity ≈ 0.7-0.85 (head only)

### 4.2 Stalk Length

- Measured from base (epidermis) to head-stalk junction
- Range: 100-500 µm for capitate-stalked
- Requires instance segmentation for accurate measurement
- Skeletonization approach: extract stalk centerline from mask

### 4.3 Density

Trichomes per unit area (mm²) — requires pixel-to-micron calibration.

- Formula: density = count / (FOV_width_mm × FOV_height_mm)
- Use stage micrometer or calibrated objective data
- Typical range: 10-200 trichomes/mm² depending on plant region and strain

---

## 5. Scientific Caveats for Computer Vision Analysis

### 5.1 What CV Analysis CAN Determine

✅ Trichome count (with >80% accuracy at 10× magnification)
✅ Trichome type classification (capitate-stalked / sessile / bulbous / non-glandular)
✅ Dominant optical stage (clear / cloudy / amber)
✅ Relative maturity comparison (this vs previous image of same plant)
✅ Density estimation (with calibration)
✅ Head diameter (with calibration, within ~10 µm accuracy)
✅ Harvest timing guidance (qualitative, strain-specific)

### 5.2 What CV Analysis CANNOT Determine

❌ THC%, CBD%, or any cannabinoid concentration
❌ Terpene profile or aroma characteristics
❌ Absolute potency comparison between strains
❌ Whether "cloudy = peak THC" (visual stage ≠ concentration)
❌ 3D trichome structure from 2D images
❌ Cell-level biochemistry

### 5.3 Known Error Sources

- Focus quality significantly affects detection rates
- Lighting inconsistency affects color-based maturity classification
- Overlapping trichomes confound instance segmentation
- Strain variation creates distribution shift between training and deployment
- Magnification must match calibration objective

---

## References

1. **Mahlberg, P.G. & Kim, E.S. (2004)**.
   Accumulation of Cannabinoids in Glandular Trichomes of Cannabis.
   J. Ind. Hemp 9(1):15-36.

2. **Fischedick, J.T. et al. (2010)**.
   Metabolic fingerprinting of Cannabis sativa L.
   Phytochemistry 71(17-18):2058-2073. DOI:10.1016/j.phytochem.2010.09.015.

3. **ElSohly, M.A. & Slade, D. (2005)**.
   Chemical constituents of marijuana.
   Life Sciences 78(5):539-548. DOI:10.1016/j.lsc.2005.09.011.

4. **Elzinga, S. et al. (2015)**.
   Cannabinoids and terpenes as chemotaxonomic markers in cannabis.
   Nat. Prod. Chem. Res. 3:181. DOI:10.4172/2329-6836.1000181.

5. **Small, E. (2015)**.
   Evolution and Classification of Cannabis sativa.
   The Botanical Review 81(3):189-294.
