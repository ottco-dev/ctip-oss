# THC → CBN Oxidative Degradation: The Amber Coloration Mechanism

## Summary

Amber trichome coloration is caused by photo-oxidative degradation of THC to CBN.
It is a **degradation signal**, not a potency indicator.

This document explains the chemistry, its visual consequences, and why automated
trichome analysis systems must not make quantitative THC claims from amber%.

---

## 1. The Oxidation Pathway

### 1.1 Molecular Structures

**THCA (Δ9-tetrahydrocannabinolic acid)**
- Non-psychoactive precursor in living plant
- Decarboxylates to THC upon heating (smoking, vaporizing)
- MW: 358.48 g/mol
- Clear, colorless when fresh

**THC (Δ9-tetrahydrocannabinol)**
- Psychoactive form; spontaneously forms from THCA
- Tricyclic terpenoid with dibenzopyranone core
- MW: 314.45 g/mol

**CBN (Cannabinol)**
- Fully aromatized degradation product of THC
- MW: 310.43 g/mol (loses 4H from THC)
- Orange-brown chromophore → **amber visual appearance**
- Mildly psychoactive (sedative)

### 1.2 Two-Step Oxidation Mechanism

```
                UV light / heat / oxygen
THCA ─────────────────────────────────► THC
                decarboxylation

                UV light / oxygen
THC  ─────────────────────────────────► Δ8-THC (isomerization)
                    Step 1

                continued oxidation
Δ8-THC ────────────────────────────────► CBN (aromatization)
                    Step 2
```

**Step 1**: Photo-isomerization
- UV radiation catalyzes migration of the Δ9 double bond to Δ8 position
- Δ8-THC is less potent but chemically more stable

**Step 2**: Aromatization → CBN
- Oxidative dehydrogenation: removal of 4H atoms (2 H₂ molecules)
- Pyran ring in THC undergoes complete aromatization
- Product: aromatic pyranochromene ring system = CBN
- CBN chromophore absorbs blue/green light → transmits orange/brown → **amber appearance**

Reference: ElSohly MA, Slade D. (2005). Life Sciences 78(5):539-548.

---

## 2. Rate Factors

The rate of THC→CBN conversion depends on:

| Factor | Effect on Degradation Rate |
|--------|---------------------------|
| UV exposure | Primary driver — direct photolysis |
| Temperature | Arrhenius relationship; each +10°C ≈ doubles rate |
| Oxygen | Required for Step 2 (aromatization) |
| Moisture | Hydrolysis may accelerate |
| Storage conditions | Dark, cool, vacuum = slowest |

**Timescale**: Stored dried cannabis at room temperature with UV exposure can
show significant THC→CBN conversion within 1-2 months.
Living plant trichomes: days to weeks during senescence.

---

## 3. Visual-Chemical Correlation (or Lack Thereof)

### 3.1 The Fundamental Decoupling

**Amber fraction (visual) ≠ THC concentration (chemical)**

This decoupling occurs because:

1. **Strain variation**: different strains produce different amounts of THCA
   in the same visual maturity stage
   - Observed: 10-20% THC difference between strains at identical visual stage
   - Reference: Elzinga et al. (2015). Nat. Prod. Chem. Res. 3:181.

2. **Individual trichome variation**: within one plant, trichomes develop asynchronously
   - 30% amber ≠ "30% of THC is gone" — different trichomes, different histories

3. **Trichome density variation**: amber% measures maturity of visible trichomes,
   not total cannabinoid pool

4. **Optical path length**: thicker trichome heads may appear amber at lower CBN%
   due to Beer-Lambert absorption path effects

### 3.2 What Amber% Does Tell Us

✅ Trichomes are undergoing oxidative degradation
✅ Harvest window is open / potentially past optimal
✅ Qualitative signal that storage conditions may need improvement
✅ Senescence signal (plant is past peak biological development)

### 3.3 What Amber% Cannot Tell Us

❌ Actual %THC or %CBD in the sample
❌ Whether potency is "high" or "low"
❌ How much THC has been lost
❌ Terpene content or aroma profile
❌ Medical efficacy

---

## 4. Correct Harvest Timing Language

The following language is **scientifically defensible**:

> "Trichome maturity analysis indicates approximately X% of observed capitate-stalked
> trichomes have entered the amber (oxidative degradation) stage. This is commonly
> associated with senescence of the resin-producing tissue. Traditional cultivation
> guidance suggests harvesting within [timeframe] to preserve terpene profile.
> Note: this does not indicate specific THC or CBD concentration."

The following language is **scientifically indefensible and should never appear in output**:

❌ "THC content is at peak"
❌ "THC level: high"
❌ "Potency: X%"
❌ "Best time to harvest for maximum THC"
❌ "CBN:THC ratio = X"

---

## 5. System Implementation Requirements

All code, UI, and report output in this system must enforce:

1. **SCIENTIFIC_CAVEAT constant**: attached to every `MaturityLabel` dataclass
2. **Harvest guidance**: use qualitative language ("may indicate", "commonly associated")
3. **No THC%/CBD% claims**: blocked at the API schema level
4. **Research citations**: include ElSohly 2005 + Elzinga 2015 in all reports
5. **Clarity on what is measured**: "optical maturity stage" not "THC content"

See `maturity/domain/scientific_rules.py` for programmatic enforcement.

---

## 6. References

1. **ElSohly, M.A. & Slade, D. (2005)**.
   Chemical constituents of marijuana: The complex mixture of natural cannabinoids.
   *Life Sciences* 78(5):539-548. DOI: 10.1016/j.lsc.2005.09.011.
   > Primary reference for THC→CBN degradation mechanism.

2. **Elzinga, S., Fischedick, J., Bassetti, R. & Raber, J.C. (2015)**.
   Cannabinoids and terpenes as chemotaxonomic markers in cannabis.
   *Natural Products Chemistry & Research* 3:181. DOI: 10.4172/2329-6836.1000181.
   > Demonstrates 10-20% THC variation between strains at identical visual maturity.

3. **Fischedick, J.T. et al. (2010)**.
   Metabolic fingerprinting of Cannabis sativa L., cannabinoids and terpenoids.
   *Phytochemistry* 71(17-18):2058-2073. DOI: 10.1016/j.phytochem.2010.09.015.
   > Explains cloudy appearance via Mie scattering / THCA crystallization.

4. **Hazekamp, A. et al. (2010)**.
   Cannabis: from cultivar to chemovar.
   *Drug Testing and Analysis* 2(5):215-222. DOI: 10.1002/dta.142.
   > Chemotaxonomic classification; supports decoupling of visual and chemical markers.

5. **Turner, C.E. et al. (1980)**.
   Constituents of cannabis sativa L.: stability of cannabinoids in stored plant material.
   *Journal of Pharmaceutical Sciences* 69(7):836-837.
   > Original degradation rate measurements in stored material.
