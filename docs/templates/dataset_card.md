# Dataset card — <name>

> Copy this file next to your dataset (`DATASET_CARD.md`) and fill in every section. A dataset without this
> information cannot be evaluated or reused fairly. Write "unknown" rather than leaving a field out.

## Summary
- **Name / version:**
- **Maintainer / contact:**
- **Licence:** (e.g. CC BY 4.0)
- **Images:** count, format, resolution
- **Purpose:** what it was collected for, and what it should *not* be used for

## Acquisition
- **Microscope / camera:** model, sensor, objective(s), adapter, digital zoom
- **Illumination:** type (LED ring, coaxial, transmitted), colour temperature, diffuser, fixed exposure / white balance?
- **Calibration:** µm per pixel, method (stage micrometer recommended), file
- **Samples:** plant material, growth stage in days, cultivar/strain *as reported* (unverified unless lab-tested)
- **Sessions:** number of imaging sessions; how images map to sessions (`data.session` in Label Studio, folder or
  file-name series). Sessions are the unit of the train/val/test split.

## Labels
- **Classes:** detection classes, morphology classes, maturity classes (clear / cloudy / amber …) with definitions
- **Tool and guideline:** Label Studio / CVAT, link to the labelling guideline
- **Annotators:** how many, experience, review process
- **Agreement:** inter-annotator agreement (Cohen's κ or IoU between annotators) on a shared subset
- **Pre-labels:** VLM or model pre-labels used? All reviewed by a person?

## Splits
- Method (CTIP session split, seed), images and sessions per split, any held-out sessions or setups

## Known issues and bias
- Lighting or focus variation, over-represented cultivars or growth stages, label noise, duplicates

## Ethics and legal
- Cultivation legality where the images were taken; no personal data in images or metadata

## Changelog
- <date>: initial release
