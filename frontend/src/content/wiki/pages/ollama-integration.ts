import type { WikiPage } from '../types';

const en = `
## Ollama Integration

CTIP uses a locally-running **Ollama** instance to generate natural-language narrative paragraphs for scientific reports. The LLM receives structured analysis results (counts, maturity distribution, morphology breakdown, measurement statistics) and produces a readable description in the selected language and style.

> **Important**: The LLM describes *optical maturity* as observed under the microscope. It does **not** claim or infer cannabinoid concentrations, THC percentages, or any pharmacological properties. Such claims are scientifically unsupported and are explicitly blocked in the prompt template.

---

## Setup

\`\`\`bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the Ollama server
ollama serve

# 3. Pull a model (3B recommended for 8 GB VRAM)
ollama pull llama3.2:3b

# 4. Verify
curl http://localhost:11434/api/tags
\`\`\`

The CTIP backend connects to \`http://localhost:11434\` by default. Override with \`OLLAMA_BASE_URL\` in \`.env\`.

---

## API reference

### Check Ollama status

\`\`\`bash
GET /api/v1/ollama/status
\`\`\`

\`\`\`json
{ "connected": true, "base_url": "http://localhost:11434", "active_model": "llama3.2:3b" }
\`\`\`

### List available models

\`\`\`bash
GET /api/v1/ollama/models
\`\`\`

### Pull a model

\`\`\`bash
POST /api/v1/ollama/models/pull
Content-Type: application/json

{ "model": "llama3.2:3b" }
\`\`\`

### Generate a narrative

\`\`\`bash
POST /api/v1/ollama/narrative
Content-Type: application/json

{
  "analysis_result": { ... },
  "style": "scientific",
  "language": "en",
  "model": "llama3.2:3b"
}
\`\`\`

Response:
\`\`\`json
{
  "narrative": "The sample displays a trichome population of 47 instances ...",
  "model": "llama3.2:3b",
  "style": "scientific",
  "language": "en",
  "generation_ms": 1840
}
\`\`\`

### Get / update config

\`\`\`bash
GET  /api/v1/ollama/config
POST /api/v1/ollama/config
\`\`\`

Config fields: \`base_url\`, \`default_model\`, \`temperature\` (0.0–1.0), \`max_tokens\`.

---

## analysis_result schema

\`\`\`json
{
  "session_id": "abc123",
  "image": "IMG_0001.tif",
  "total_trichomes": 47,
  "morphology": {
    "capitate_stalked": 31,
    "capitate_sessile": 12,
    "bulbous": 3,
    "non_glandular": 1
  },
  "maturity": { "clear": 8, "cloudy": 35, "amber": 4 },
  "measurements": {
    "mean_size_um": 23.1,
    "std_size_um": 4.7,
    "min_size_um": 11.2,
    "max_size_um": 38.9
  },
  "calibration": "DeltaOptical-40x.json",
  "model": "yolo11s_custom.pt",
  "inference_time_ms": 234
}
\`\`\`

---

## Narrative styles

| Style | Audience | Tone |
|---|---|---|
| \`scientific\` | Researchers, lab reports | Precise, passive voice, includes uncertainty |
| \`summary\` | General users, quick overview | Plain language, no jargon |
| \`technical\` | Engineers, QC workflows | Metric-focused, structured lists |

Supported languages: \`en\` (English), \`de\` (German), \`es\` (Spanish).

---

## Frontend: Reports page → AI Narrative panel

- **Model selector** — dropdown of pulled models.
- **Style radio** — scientific / summary / technical.
- **Language selector** — EN / DE / ES.
- **Generate button** — streams response tokens.
- **Narrative text area** — editable after generation; included in PDF export.
- **Status indicator** — shows Ollama connection health.
`;

const de = `
## Ollama-Integration

CTIP verwendet eine lokal laufende **Ollama**-Instanz, um natürlichsprachliche Narrative für wissenschaftliche Berichte zu erstellen. Das LLM erhält strukturierte Analyseergebnisse und erzeugt eine lesbare Beschreibung in der gewählten Sprache und im gewählten Stil.

> **Wichtig**: Das LLM beschreibt ausschließlich die *optische Reife* unter dem Mikroskop. Es macht **keine** Aussagen über Cannabinoid-Konzentrationen, THC-Gehalt oder pharmakologische Eigenschaften.

---

## Setup

\`\`\`bash
# Ollama installieren und starten
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3.2:3b
\`\`\`

---

## API-Referenz

\`\`\`bash
GET  /api/v1/ollama/status
GET  /api/v1/ollama/models
POST /api/v1/ollama/models/pull
POST /api/v1/ollama/narrative
GET  /api/v1/ollama/config
POST /api/v1/ollama/config
\`\`\`

---

## Narrativ-Stile

| Stil | Zielgruppe | Ton |
|---|---|---|
| \`scientific\` | Forscher, Laborberichte | Präzise, Passiv, enthält Unsicherheit |
| \`summary\` | Allgemeine Nutzer | Einfache Sprache, kein Fachjargon |
| \`technical\` | Ingenieure, QS-Workflows | Metrikfokussiert, strukturierte Listen |

Unterstützte Sprachen: Englisch, Deutsch, Spanisch.

---

## Frontend: Berichte-Seite → KI-Narrativ-Panel

- **Modell-Auswahl** — Dropdown der heruntergeladenen Modelle.
- **Stil-Auswahl** — wissenschaftlich / Zusammenfassung / technisch.
- **Sprach-Auswahl** — EN / DE / ES.
- **Generieren-Schaltfläche** — streamt Antwort-Token.
- **Narrativ-Textfeld** — nach der Generierung bearbeitbar; im PDF-Export enthalten.
`;

const es = `
## Integración con Ollama

CTIP usa una instancia local de **Ollama** para generar narrativas en lenguaje natural para informes científicos. El LLM recibe resultados estructurados y produce una descripción legible en el idioma y estilo seleccionados.

> **Importante**: El LLM describe únicamente la *madurez óptica* observada bajo el microscopio. No afirma ni infiere concentraciones de cannabinoides, porcentajes de THC ni propiedades farmacológicas.

---

## Configuración

\`\`\`bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3.2:3b
\`\`\`

---

## Referencia de API

\`\`\`bash
GET  /api/v1/ollama/status
GET  /api/v1/ollama/models
POST /api/v1/ollama/models/pull
POST /api/v1/ollama/narrative
GET  /api/v1/ollama/config
POST /api/v1/ollama/config
\`\`\`

---

## Estilos de narrativa

| Estilo | Audiencia | Tono |
|---|---|---|
| \`scientific\` | Investigadores, informes de laboratorio | Preciso, voz pasiva, incluye incertidumbre |
| \`summary\` | Usuarios generales | Lenguaje simple, sin jerga |
| \`technical\` | Ingenieros, flujos QC | Enfocado en métricas |

Idiomas soportados: inglés, alemán, español.
`;

const page: WikiPage = {
  slug: 'ollama-integration',
  title: {
    en: 'Ollama Integration',
    de: 'Ollama-Integration',
    es: 'Integración con Ollama',
  },
  description: {
    en: 'Local LLM for scientific report narrative generation. Describes optical maturity only — no cannabinoid claims.',
    de: 'Lokales LLM zur Erstellung wissenschaftlicher Berichtsnarrative. Beschreibt nur optische Reife — keine Cannabinoid-Behauptungen.',
    es: 'LLM local para narrativas de informes científicos. Solo describe madurez óptica — sin afirmaciones sobre cannabinoides.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '🤖',
};

export default page;
