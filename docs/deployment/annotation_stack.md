# Annotation Stack — Label Studio + PostgreSQL

## Overview

The annotation stack provides a Label Studio instance backed by PostgreSQL for
persistent annotation storage. This is the primary human-in-the-loop (HITL) tool
for trichome image annotation.

**Services:**
- `label-studio`: Annotation platform (host port **3005**, container 8080)
- `ls-postgres`: PostgreSQL 15 database (host port **3007**, container 5432)

**Public access (via nginx):** `http://ottco.ddns.net:3001/annotation/`

**Hardware:** CPU-only. No GPU required for annotation.

---

## Quick Start

```bash
# 1. Copy environment template
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and LABEL_STUDIO_SECRET_KEY

# 2. Start annotation stack via profile (preferred — shares trichome-net with main stack)
cd docker
docker compose --profile annotation up -d

# 3. Wait for startup (~60s)
docker compose logs -f label-studio

# 4. Open http://localhost:3005  (or via nginx: http://ottco.ddns.net:3001/annotation/)
# Create admin account on first visit
```

---

## Environment Variables

Add to `.env`:

```env
# Annotation stack
LABEL_STUDIO_SECRET_KEY=CHANGE_THIS_IN_PRODUCTION_USE_LONG_RANDOM_STRING
LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=false

# PostgreSQL (internal — port 3007 on host)
POSTGRES_USER=labelstudio
POSTGRES_PASSWORD=CHANGE_THIS_IN_PRODUCTION
POSTGRES_DB=labelstudio

# Label Studio (set after first login)
# Direct access:   http://localhost:3005
# Via nginx proxy: http://ottco.ddns.net:3001/annotation/
LABEL_STUDIO_API_KEY=<copy from Account Settings → Access Token>
LABEL_STUDIO_URL=http://localhost:3005
```

---

## Trichome Annotation Configuration

### Labeling Interface (XML Config)

Create a new project in Label Studio with this labeling interface:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>

  <Header value="Trichome Type"/>
  <RectangleLabels name="bbox" toName="image">
    <Label value="capitate_stalked"  background="#FF6B6B"/>
    <Label value="capitate_sessile"  background="#FFA94D"/>
    <Label value="bulbous"           background="#FFD43B"/>
    <Label value="non_glandular"     background="#74C0FC"/>
  </RectangleLabels>

  <Header value="Maturity Stage (Optical Observation Only)"/>
  <PolygonLabels name="mask" toName="image">
    <Label value="clear"    background="#A9E34B"/>
    <Label value="cloudy"   background="#FFFFFF"/>
    <Label value="amber"    background="#FFA94D"/>
    <Label value="degraded" background="#845EF7"/>
  </PolygonLabels>

  <Header value="Image Quality"/>
  <Choices name="quality" toName="image" choice="single">
    <Choice value="good"/>
    <Choice value="acceptable"/>
    <Choice value="poor"/>
  </Choices>

  <TextArea name="notes" toName="image" placeholder="Reviewer notes (optional)"
            maxSubmissions="1" editable="true" rows="2"/>
</View>
```

### Scientific Annotation Guidelines

**Trichome classification (morphology):**
- `capitate_stalked`: Elongated trichome with distinct stalk + spherical head. 100–500µm total height. Most common glandular type.
- `capitate_sessile`: Short/absent stalk, head sits directly on leaf surface. 25–100µm head diameter.
- `bulbous`: Very small, round, non-stalked, ~10-30µm. Often requires 40×+ objective.
- `non_glandular`: Elongated/hair-like, no secretory head. Excludes from cannabinoid analysis.

**Maturity stage (optical observation):**
- `clear`: Glassy, transparent head. Light passes through cleanly.
- `cloudy`: Opaque, white/milky head. Light scatters instead of transmitting.
- `amber`: Warm golden/amber hue. Indicates oxidative color shift.
- `degraded`: Brown to dark brown. Collapsed or burst head structures.

⚠️ **CRITICAL SCIENTIFIC NOTE**: Maturity stage describes OPTICAL COLOR STATE only.
Do NOT annotate "THC content", "potency", or "harvest readiness". These are inference
claims beyond the scope of visual observation.

---

## Integration with Trichome Platform

### Export annotations to training dataset

```bash
# Via CLI
trichome annotate stats --label-studio-url http://localhost:3005

# Via API (backend port 3002 on host)
curl http://localhost:3002/api/v1/annotation/label-studio/export/1 \
  -H "X-API-Key: $TRICHOME_API_KEY"
```

### Import detection results for review

```bash
# Push YOLO predictions to Label Studio for review
trichome annotate run \
  --input-dir data/microscopy/ \
  --output-dir data/pending_review/ \
  --label-studio-url http://localhost:3005
```

### HITL policy

All VLM-generated labels **must** pass through Label Studio review before
entering the training dataset. This is enforced in:
- `apps/cli/commands/annotate.py` (always shows HITL notice)
- `vlm_labeling/application/auto_label_pipeline.py` (pending_review queue)
- `annotation/review_queue.py` (approval required)

---

## Running with Full Dev Stack

```bash
# Preferred: annotation services are defined as profile in docker-compose.yml
cd docker
docker compose --profile annotation up -d

# Service URLs:
# nginx (public entry):  http://ottco.ddns.net:3001
# Backend API:           http://localhost:3002/api/v1
# Frontend:              http://localhost:3003
# Label Studio:          http://localhost:3005  (or /annotation/ via nginx)
# MLflow:                http://localhost:3004  (or /mlflow/ via nginx)
# CVAT:                  http://localhost:3006  (or /cvat/ via nginx)
```

---

## Data Volumes

| Volume | Content | Backup priority |
|---|---|---|
| `label-studio-data` | Projects, exports, media files | **HIGH** |
| `label-studio-pg` | PostgreSQL annotations database | **CRITICAL** |

### Backup

```bash
# Backup PostgreSQL
docker exec trichome-ls-postgres pg_dump \
  -U labelstudio labelstudio > backup_annotations_$(date +%Y%m%d).sql

# Backup Label Studio data
docker run --rm \
  -v label-studio-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/ls-data-$(date +%Y%m%d).tar.gz /data
```

### Restore

```bash
# Restore PostgreSQL
docker exec -i trichome-ls-postgres psql \
  -U labelstudio labelstudio < backup_annotations_20260101.sql
```

---

## Troubleshooting

**Label Studio fails to start:**
```bash
# Check postgres health
docker logs trichome-ls-postgres

# Check if port 3005 is already in use
lsof -i :3005
ss -tlnp | grep 3005
```

**Cannot connect to database:**
- Verify `POSTGRES_PASSWORD` matches in both `ls-postgres` and `label-studio` service env vars
- Check postgres healthcheck: `docker inspect trichome-ls-postgres`

**Large image uploads fail:**
- `DATA_UPLOAD_MAX_MEMORY_SIZE` is set to 512MB by default
- For 16-bit TIFF stacks > 512MB, mount via local filesystem storage backend instead of upload

**HITL bypass detected:**
- Check `vlm_labeling/application/auto_label_pipeline.py` — `_write_to_pending_review()` must never be bypassed
- The review queue in `annotation/review_queue.py` is the single source of truth
