import type { WikiPage } from '../types';

const en = `
## VLM Ensemble

Instead of relying on a single Vision-Language Model for auto-labeling, CTIP can run **multiple VLM backends simultaneously** and aggregate their predictions via majority vote. Ensemble mode increases label reliability and surfaces disagreement as an explicit quality signal.

> **Invariant**: VLM outputs — whether from a single model or an ensemble — are **never** written directly to the training dataset. Every label is placed in the \`pending_review\` queue and must be approved by a human annotator before it becomes training data.

---

## Supported backends

| Backend | Size | VRAM (4-bit quant) | Notes |
|---|---|---|---|
| Moondream-2B | 2B | ~1.8 GB | Fastest; good for binary decisions |
| Florence-2 | 0.23B–0.77B | ~0.8 GB | Strong grounding and counting |
| Qwen2-VL | 2B–7B | ~2.4 GB (2B) | Best general VQA accuracy |

All models run 4-bit quantized to stay within the 8 GB VRAM budget.

---

## Agreement scoring

\`\`\`
agreement = count(majority_label) / total_models
\`\`\`

| Score | Level | Action |
|---|---|---|
| ≥ 0.8 | High | Label sent to \`pending_review\` with high confidence |
| ≥ 0.6 | Medium | Label sent to \`pending_review\`; reviewer notified |
| < 0.6 | Low | Label flagged; reviewer must adjudicate before approval |

---

## Prompt system

CTIP ships preset prompt templates optimised for trichome labeling tasks. You can also supply custom \`system\` and \`user\` prompts per request.

### Preset IDs

| Preset | Task |
|---|---|
| \`morphology_classify\` | Classify trichome type (stalked / sessile / bulbous / non-glandular) |
| \`maturity_classify\` | Classify maturity stage (clear / cloudy / amber) |
| \`count_estimate\` | Estimate trichome count in a region |
| \`quality_check\` | Assess image quality (focus, lighting, artifacts) |

---

## API reference

### Run ensemble labeling

\`\`\`bash
POST /api/v1/vlm/ensemble/label
Content-Type: application/json

{
  "image_id": "img_0047",
  "models": ["moondream", "florence2", "qwen2vl"],
  "preset": "morphology_classify",
  "system_prompt_override": null,
  "user_prompt_override": null
}
\`\`\`

Response:
\`\`\`json
{
  "image_id": "img_0047",
  "majority_label": "CAPITATE_STALKED",
  "agreement_score": 0.83,
  "agreement_level": "high",
  "per_model": {
    "moondream":  { "label": "CAPITATE_STALKED", "confidence": 0.91 },
    "florence2":  { "label": "CAPITATE_STALKED", "confidence": 0.87 },
    "qwen2vl":    { "label": "CAPITATE_SESSILE",  "confidence": 0.61 }
  },
  "review_id": "rev_9f3c"
}
\`\`\`

### List prompt presets

\`\`\`bash
GET /api/v1/vlm/prompts
\`\`\`

### Create / update a custom prompt

\`\`\`bash
POST /api/v1/vlm/prompts
Content-Type: application/json

{
  "id": "my_custom_preset",
  "description": "Custom morphology prompt for lab X",
  "system_prompt": "...",
  "user_prompt": "..."
}
\`\`\`

### Validate a prompt

\`\`\`bash
POST /api/v1/vlm/prompts/validate
Content-Type: application/json

{
  "system_prompt": "...",
  "user_prompt": "...",
  "test_image_id": "img_0001"
}
\`\`\`

Returns a dry-run response showing what each enabled backend would return, without writing to the review queue.

---

## Frontend: Annotation page → VLM Config panel

- **Ensemble mode toggle** — enable to select multiple backends; disable to use a single model.
- **Model checkboxes** — select which backends to include; VRAM estimate shown per combination.
- **Preset dropdown** — choose a labeling task; preview of system/user prompts displayed.
- **Custom prompt editor** — override system and/or user prompt for the current run.
- **Agreement threshold slider** — minimum score for automatic forwarding to the review queue.
- **Results panel** — per-model labels, confidence scores, majority label badge, agreement level indicator.

---

## Agreement visualisation

| Colour | Level | Threshold |
|---|---|---|
| Green | High | ≥ 0.8 |
| Yellow | Medium | ≥ 0.6 |
| Red | Low | < 0.6 |
`;

const de = `
## VLM-Ensemble

Statt eines einzelnen Vision-Language-Modells kann CTIP **mehrere VLM-Backends gleichzeitig** ausführen und deren Vorhersagen per Mehrheitsvoting aggregieren.

> **Invariante**: VLM-Ausgaben werden **niemals** direkt in den Trainingsdatensatz geschrieben. Jedes Label muss von einem menschlichen Annotator genehmigt werden.

---

## Unterstützte Backends

| Backend | Größe | VRAM (4-Bit) |
|---|---|---|
| Moondream-2B | 2B | ~1,8 GB |
| Florence-2 | 0,23B–0,77B | ~0,8 GB |
| Qwen2-VL | 2B–7B | ~2,4 GB (2B) |

---

## Übereinstimmungsbewertung

\`\`\`
agreement = Anzahl(Mehrheitslabel) / Gesamtmodelle
\`\`\`

| Wert | Stufe | Aktion |
|---|---|---|
| ≥ 0,8 | Hoch | Label mit hoher Konfidenz an \`pending_review\` |
| ≥ 0,6 | Mittel | Label an \`pending_review\`, Reviewer benachrichtigt |
| < 0,6 | Niedrig | Label markiert; Schiedsrichter erforderlich |

---

## API-Referenz

\`\`\`bash
POST /api/v1/vlm/ensemble/label
GET  /api/v1/vlm/prompts
POST /api/v1/vlm/prompts
POST /api/v1/vlm/prompts/validate
\`\`\`

---

## Frontend: Annotations-Seite → VLM-Konfigurationsfeld

- **Ensemble-Modus** — mehrere Backends auswählen.
- **Modell-Checkboxen** — VRAM-Schätzung pro Kombination.
- **Preset-Dropdown** — Aufgabe auswählen; Prompt-Vorschau anzeigen.
- **Benutzerdefin. Prompt-Editor** — System-/Nutzer-Prompt überschreiben.
- **Ergebnisfeld** — Modell-Labels, Konfidenz, Mehrheitslabel, Übereinstimmungsstufe.
`;

const es = `
## Ensemble VLM

En lugar de depender de un único modelo de visión-lenguaje, CTIP puede ejecutar **múltiples backends VLM simultáneamente** y agregar sus predicciones por votación mayoritaria.

> **Invariante**: Las salidas de VLM **nunca** se escriben directamente en el dataset de entrenamiento. Cada etiqueta debe ser aprobada por un anotador humano.

---

## Backends soportados

| Backend | Tamaño | VRAM (quant. 4-bit) |
|---|---|---|
| Moondream-2B | 2B | ~1.8 GB |
| Florence-2 | 0.23B–0.77B | ~0.8 GB |
| Qwen2-VL | 2B–7B | ~2.4 GB (2B) |

---

## Puntuación de acuerdo

\`\`\`
agreement = count(etiqueta_mayoritaria) / total_modelos
\`\`\`

| Puntuación | Nivel | Acción |
|---|---|---|
| ≥ 0.8 | Alto | Etiqueta a \`pending_review\` con alta confianza |
| ≥ 0.6 | Medio | Etiqueta a \`pending_review\`, revisor notificado |
| < 0.6 | Bajo | Etiqueta marcada; se requiere arbitraje |

---

## Referencia de API

\`\`\`bash
POST /api/v1/vlm/ensemble/label
GET  /api/v1/vlm/prompts
POST /api/v1/vlm/prompts
POST /api/v1/vlm/prompts/validate
\`\`\`

---

## Frontend: página de Anotación → panel VLM Config

- **Toggle de modo Ensemble** — habilitar para usar múltiples backends.
- **Casillas de modelos** — seleccionar backends; estimación de VRAM por combinación.
- **Menú de presets** — elegir tarea; vista previa de prompts.
- **Editor de prompt personalizado** — sobreescribir prompts de sistema/usuario.
- **Panel de resultados** — etiquetas por modelo, confianza, etiqueta mayoritaria, indicador de nivel de acuerdo.
`;

const page: WikiPage = {
  slug: 'vlm-ensemble',
  title: {
    en: 'VLM Ensemble',
    de: 'VLM-Ensemble',
    es: 'Ensemble VLM',
  },
  description: {
    en: 'Run multiple VLM providers simultaneously with majority-vote consensus for reliable auto-labeling.',
    de: 'Mehrere VLM-Backends gleichzeitig mit Mehrheitsvoting für zuverlässiges Auto-Labeling.',
    es: 'Ejecutar múltiples backends VLM simultáneamente con consenso por votación mayoritaria.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '🗳️',
};

export default page;
