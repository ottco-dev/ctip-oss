## Ollama Integration

CTIP uses a locally-running **Ollama** instance to generate natural-language narrative paragraphs for scientific reports. The LLM is given structured analysis results (counts, maturity distribution, morphology breakdown, measurement statistics) and produces a readable description in the selected language and style.

> **Important**: The LLM describes *optical maturity* as observed under the microscope. It does **not** claim or infer cannabinoid concentrations, THC percentages, or any pharmacological properties. Such claims are scientifically unsupported and are explicitly blocked in the prompt template.

---

## Setup

```bash
# 1. Install Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the Ollama server
ollama serve

# 3. Pull a model (3B parameter model recommended for 8 GB VRAM)
ollama pull llama3.2:3b

# 4. Verify
curl http://localhost:11434/api/tags
```

The CTIP backend connects to `http://localhost:11434` by default. Override with `OLLAMA_BASE_URL` in `.env`.

---

## API reference

### Check Ollama status

```bash
GET /api/v1/ollama/status
```

```json
{
  "connected": true,
  "base_url": "http://localhost:11434",
  "active_model": "llama3.2:3b"
}
```

### List available models

```bash
GET /api/v1/ollama/models
```

```json
{
  "models": [
    { "name": "llama3.2:3b", "size_gb": 2.0, "modified_at": "2025-03-01" }
  ]
}
```

### Pull a model

```bash
POST /api/v1/ollama/models/pull
Content-Type: application/json

{ "model": "llama3.2:3b" }
```

Streams pull progress via the response body.

### Generate a narrative

```bash
POST /api/v1/ollama/narrative
Content-Type: application/json

{
  "analysis_result": { ... },
  "style": "scientific",
  "language": "en",
  "model": "llama3.2:3b"
}
```

Response:
```json
{
  "narrative": "The sample displays a trichome population of 47 instances ...",
  "model": "llama3.2:3b",
  "style": "scientific",
  "language": "en",
  "generation_ms": 1840
}
```

### Get / update Ollama config

```bash
GET  /api/v1/ollama/config
POST /api/v1/ollama/config
```

Config fields: `base_url`, `default_model`, `temperature` (0.0–1.0), `max_tokens`.

---

## analysis_result schema

The JSON sent to the LLM contains:

```json
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
  "maturity": {
    "clear": 8,
    "cloudy": 35,
    "amber": 4
  },
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
```

---

## Narrative styles

| Style | Audience | Tone |
|---|---|---|
| `scientific` | Researchers, lab reports | Precise, passive voice, includes uncertainty |
| `summary` | General users, quick overview | Plain language, no jargon |
| `technical` | Engineers, QC workflows | Metric-focused, structured lists |

Supported languages: `en` (English), `de` (German), `es` (Spanish).

---

## Frontend: Reports page → AI Narrative panel

- **Model selector** — dropdown of pulled models from `/api/v1/ollama/models`.
- **Style radio** — scientific / summary / technical.
- **Language selector** — EN / DE / ES.
- **Generate button** — calls `/api/v1/ollama/narrative`; streams response tokens.
- **Narrative text area** — editable after generation; included in PDF export.
- **Status indicator** — shows Ollama connection health from `/api/v1/ollama/status`.
