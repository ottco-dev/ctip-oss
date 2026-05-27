import type { WikiPage } from '../types';

const en = `
## Label Studio overview

Label Studio (:3005) is the primary annotation platform for CTIP.
It handles image labeling, review workflows, and export to training format.

Start it:
\`\`\`bash
cd docker && docker compose --profile annotation up -d
# or from setup wizard: Step 5 → Start Annotation Stack
\`\`\`

Access: http://localhost:3005

---

## Trichome classes

| Class | Color | Hotkey | Description |
|-------|-------|--------|-------------|
| **stalked** | cyan \`#22d3ee\` | 1 | Glandular trichome with visible stalk + head. Most common in mature flowers. |
| **sessile** | green \`#34d399\` | 2 | Short or no stalk, head sits close to the surface. |
| **bulbous** | purple \`#a78bfa\` | 3 | Smallest type, visible only under high magnification. |
| **non-glandular** | orange \`#fb923c\` | 4 | Hair-like, no resin head. |

**Quality rating (per image):**
- \`good\` — sharp, well-lit, all trichomes clearly visible
- \`blurry\` — focus issues, trichomes indistinct
- \`poor\` — overexposed, underexposed, or severe artifacts

**Notes field**: Free text for session notes (e.g., "day 65 flowering", "possible mold").

---

## Annotation workflow

### 1. Open a task

In Label Studio → project → task list → click any image.
Keyboard: \`d\` next, \`a\` previous, \`w\` submit.

### 2. Draw bounding boxes

1. Press hotkey (1–4) to select a class
2. Click and drag to draw a rectangle around a trichome head
3. Box snaps to visible area — include the full head
4. Repeat for every trichome in the image

### 3. Quality rating

After labeling all boxes:
- Click the appropriate quality choice at the bottom
- Add notes if relevant

### 4. Submit

Click **Submit** (or press \`w\`).

---

## Quality guidelines

### What to include
- Every trichome head fully visible in frame
- Partially cut trichomes at image edges (if >50% visible)

### What to exclude
- Trichomes completely out of focus
- Trichomes where class is genuinely ambiguous

### Box sizing
- Draw tight around the trichome **head** (not the stalk)
- Include the head + a 2-3px margin
- Don't include neighboring trichomes in the same box

### Common mistakes
| Mistake | Consequence |
|---------|-------------|
| Skipping small trichomes | Model misses small ones → lower recall |
| Boxes too loose | Model learns incorrect spatial extent |
| Wrong class for ambiguous trichomes | Class confusion in model |
| Inconsistent quality across sessions | Inconsistent training signal |

---

## Maturity stages

CTIP classifies maturity optically based on head color/translucency:

| Stage | Visual | Description |
|-------|--------|-------------|
| **clear** | Transparent/glassy | Early development. Resin glands not fully developed. |
| **cloudy** | Milky white, opaque | Peak development. Most common harvest target. |
| **amber** | Yellow/amber tint | Degradation beginning. |

> **Scientific note**: Color alone does not determine cannabinoid content. Maturity is an optical proxy only. Multiple factors (genetics, lighting, microscope) affect color perception.

Maturity is **not annotated in Label Studio** — it is predicted by the maturity classifier after detection.

---

## Export to training format

\`\`\`bash
# Export completed annotations as YOLO format
trichome export --project 1 --format yolo --output data/datasets/strain-A-v1/

# Structure created:
# data/datasets/strain-A-v1/
# ├── images/
# │   ├── train/
# │   └── val/
# ├── labels/
# │   ├── train/
# │   └── val/
# └── data.yaml
\`\`\`

\`\`\`yaml
# data.yaml
path: /home/user/ctip-oss/data/datasets/strain-A-v1
train: images/train
val: images/val
nc: 4
names: [stalked, sessile, bulbous, non-glandular]
\`\`\`

---

## Active learning

CTIP tracks model confidence on unlabeled images and surfaces the most uncertain ones:

\`\`\`bash
# Run inference on unlabeled pool
trichome active-learn --pool data/raw/unlabeled/ --model yolo11s --strategy entropy

# Exports ranked list to Label Studio as new tasks
# Annotate these first for maximum training value
\`\`\`

---

## CVAT (alternative annotation)

CVAT (:3006) is available for video annotation or when polygon/polyline labeling is needed.

\`\`\`bash
# Start CVAT alongside Label Studio
docker compose --profile annotation up -d

# Access
http://localhost:3006

# Default credentials (first launch)
# User: admin   Password: admin  (CHANGE IMMEDIATELY)
\`\`\`

CVAT export → YOLO format → same training pipeline as Label Studio.
`;

const de = `
## Label Studio Übersicht

Label Studio (:3005) ist die primäre Annotations-Plattform für CTIP.

Starten:
\`\`\`bash
cd docker && docker compose --profile annotation up -d
\`\`\`

Zugriff: http://localhost:3005

---

## Trichom-Klassen

| Klasse | Farbe | Hotkey | Beschreibung |
|--------|-------|--------|--------------|
| **stalked** | Cyan | 1 | Drüsentrichom mit sichtbarem Stiel + Kopf |
| **sessile** | Grün | 2 | Kurzer oder kein Stiel, Kopf nahe der Oberfläche |
| **bulbous** | Lila | 3 | Kleinster Typ, nur bei starker Vergrößerung sichtbar |
| **non-glandular** | Orange | 4 | Haarförmig, kein Harzkopf |

**Qualitätsbewertung pro Bild:** \`good\` / \`blurry\` / \`poor\`

**Notizen-Feld**: Freitext für Session-Notizen.

---

## Annotations-Workflow

1. **Task öffnen** → Bild in der Aufgabenliste anklicken
2. **Bounding Boxes zeichnen**: Hotkey (1–4) → Rechteck um Trichomkopf ziehen
3. **Qualität bewerten**: good / blurry / poor auswählen
4. **Submit** (oder \`w\` drücken)

Tastenkürzel: \`d\` nächstes, \`a\` vorheriges, \`w\` einreichen.

---

## Qualitätsrichtlinien

**Einschließen:**
- Jeden vollständig sichtbaren Trichomkopf
- Randtrichome (wenn >50% sichtbar)

**Ausschließen:**
- Komplett unscharfe Trichome
- Trichome mit genuiner Klassenambiguität

**Box-Größe:** Eng um den **Kopf** zeichnen (nicht den Stiel), 2-3px Rand.

### Häufige Fehler

| Fehler | Konsequenz |
|--------|-----------|
| Kleine Trichome überspringen | Modell erkennt kleine → niedrigerer Recall |
| Zu lockere Boxen | Modell lernt falsche räumliche Ausdehnung |
| Falsche Klasse bei ambiguosen Trichomen | Klassenverwirrung im Modell |

---

## Reifestadien (zur Information)

| Stadium | Erscheinung | Beschreibung |
|---------|-------------|--------------|
| **clear** | Transparent/glasig | Frühe Entwicklung |
| **cloudy** | Milchweiß, opak | Höchste Entwicklung |
| **amber** | Gelb/Bernstein | Beginnende Degradierung |

> **Wissenschaftlicher Hinweis**: Farbe allein bestimmt nicht den Cannabinoidgehalt. Reifegrad ist ein rein optischer Indikator.

Reifegrad wird **nicht in Label Studio annotiert** — er wird nach der Erkennung vom Maturity-Classifier vorhergesagt.

---

## Export in Trainingsformat

\`\`\`bash
trichome export --project 1 --format yolo --output data/datasets/strain-A-v1/
\`\`\`

Erstellt YOLO-Verzeichnisstruktur mit images/, labels/ und data.yaml.

---

## Active Learning

CTIP identifiziert Bilder mit geringster Modellkonfidenz:

\`\`\`bash
trichome active-learn --pool data/raw/unlabeled/ --model yolo11s --strategy entropy
\`\`\`

Diese zuerst annotieren für maximalen Trainingsgewinn.
`;

const es = `
## Label Studio

Label Studio (:3005) es la plataforma principal de anotación.

\`\`\`bash
cd docker && docker compose --profile annotation up -d
# http://localhost:3005
\`\`\`

## Clases de tricomas

| Clase | Color | Atajo | Descripción |
|-------|-------|-------|-------------|
| **stalked** | Cian | 1 | Tricoma glandular con tallo + cabeza visible |
| **sessile** | Verde | 2 | Tallo corto, cabeza cerca de la superficie |
| **bulbous** | Púrpura | 3 | Tipo más pequeño, solo visible a alta magnificación |
| **non-glandular** | Naranja | 4 | Con forma de cabello, sin cabeza de resina |

## Flujo de trabajo de anotación

1. Abrir tarea → seleccionar clase (tecla 1–4)
2. Dibujar rectángulo alrededor de la cabeza del tricoma
3. Repetir para todos los tricomas visibles
4. Seleccionar calidad: \`good\` / \`blurry\` / \`poor\`
5. Submit (\`w\`)

## Exportar a formato de entrenamiento

\`\`\`bash
trichome export --project 1 --format yolo --output data/datasets/cepa-A-v1/
\`\`\`

## Active Learning

\`\`\`bash
trichome active-learn --pool data/raw/sin-etiquetar/ --model yolo11s --strategy entropy
\`\`\`

Identifica las imágenes con menor confianza del modelo — anotar primero para mayor ganancia de entrenamiento.
`;

const page: WikiPage = {
  slug: 'labeling',
  title: { en: 'Labeling & Annotation', de: 'Labeling & Annotation', es: 'Etiquetado & Anotación' },
  description: {
    en: 'Label Studio workflow, trichome classes, quality guidelines, and export.',
    de: 'Label Studio Workflow, Trichom-Klassen, Qualitätsrichtlinien und Export.',
    es: 'Flujo de Label Studio, clases de tricomas, guías de calidad y exportación.',
  },
  content: { en, de, es },
  section: 'workflow',
  icon: '🏷️',
};

export default page;
