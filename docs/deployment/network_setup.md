# Network Setup — Trichome Analysis Platform

Public entry point: **http://your-domain.com:3001**

---

## Port Layout (3001–3010)

| Host Port | Service          | Container Port | Public Path (via nginx)          |
|-----------|-----------------|----------------|----------------------------------|
| **3001**  | nginx (proxy)   | 80             | — (entry point)                  |
| **3002**  | FastAPI backend | 8000           | `/api/v1/`, `/ws/`               |
| **3003**  | Next.js frontend| 3000           | `/`                              |
| **3004**  | MLflow          | 5000           | `/mlflow/`                       |
| **3005**  | Label Studio    | 8080           | `/annotation/`                   |
| **3006**  | CVAT            | 8080           | `/cvat/`                         |
| **3007**  | PostgreSQL      | 5432           | — (internal, not public)         |
| **3008–3010** | Reserved   | —              | —                                |

All inter-service traffic stays inside the `trichome-net` Docker bridge network.
Only **port 3001** is forwarded from the router.

---

## DDNS Setup (your-domain.com)

The DDNS hostname `your-domain.com` must always resolve to this machine's public IP.

### Option A — No-IP client (headless)

```bash
# Install
sudo apt install ddclient

# /etc/ddclient.conf
protocol=noip
use=web
server=dynupdate.no-ip.com
login=your_noip_email
password=your_noip_password
your-domain.com

# Enable and start
sudo systemctl enable ddclient
sudo systemctl start ddclient
sudo systemctl status ddclient
```

### Option B — ddclient with No-IP via Docker

```bash
docker run -d \
  --name=ddclient \
  --restart=unless-stopped \
  -e PUID=1000 -e PGID=1000 \
  -e TZ=America/Anchorage \
  -v /path/to/ddclient/config:/config \
  lscr.io/linuxserver/ddclient:latest
```

### Option C — Cron-based curl update

```bash
# Add to crontab (every 5 min):
*/5 * * * * curl -s "https://dynupdate.no-ip.com/nic/update?hostname=your-domain.com" \
  -u "email:password" >> /var/log/ddns.log 2>&1
```

### Verify resolution

```bash
nslookup your-domain.com
dig your-domain.com A +short
```

---

## Router Port Forwarding

Forward **TCP port 3001** from your router's WAN interface to this machine's LAN IP.

Example (most routers):

| Rule | Protocol | External Port | Internal IP   | Internal Port |
|------|----------|---------------|---------------|---------------|
| CTIP | TCP      | 3001          | 192.168.x.y   | 3001          |

Find your LAN IP:

```bash
ip route get 1.1.1.1 | awk '{print $7}' | head -1
# or
hostname -I | awk '{print $1}'
```

> **Security note**: Only expose port 3001. Never forward ports 3002–3007 directly.
> MLflow and annotation tools have no built-in authentication — keep them behind nginx.

---

## Starting the Stack

### Core stack (nginx + backend + frontend + MLflow)

```bash
cd /path/to/trichome-analysis/docker
docker compose up -d
```

### With annotation tools (adds Label Studio + CVAT + PostgreSQL)

```bash
cd /path/to/trichome-analysis/docker
docker compose --profile annotation up -d
```

### With training stack

```bash
cd /path/to/trichome-analysis/docker
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d
```

---

## Access URLs

| Service       | Local (direct)                   | Via Nginx (LAN)                        | Via DDNS (public)                        |
|---------------|----------------------------------|----------------------------------------|------------------------------------------|
| Frontend      | http://localhost:3003            | http://localhost:3001/                 | http://your-domain.com:3001/              |
| Backend API   | http://localhost:3002/api/v1     | http://localhost:3001/api/v1/          | http://your-domain.com:3001/api/v1/       |
| Backend WS    | ws://localhost:3002/ws           | ws://localhost:3001/ws/                | ws://your-domain.com:3001/ws/             |
| MLflow        | http://localhost:3004            | http://localhost:3001/mlflow/          | http://your-domain.com:3001/mlflow/       |
| Label Studio  | http://localhost:3005            | http://localhost:3001/annotation/      | http://your-domain.com:3001/annotation/   |
| CVAT          | http://localhost:3006            | http://localhost:3001/cvat/            | http://your-domain.com:3001/cvat/         |

---

## Health Checks

```bash
# Nginx proxy health
curl http://localhost:3001/health

# Backend API health
curl http://localhost:3001/api/v1/system/health
curl http://localhost:3002/api/v1/system/health

# MLflow
curl http://localhost:3004/health

# Label Studio
curl http://localhost:3005/health

# Check all containers
docker compose ps
docker compose logs --tail=50 nginx
```

---

## Nginx Configuration

The nginx config is at `docker/nginx/nginx.conf` and is mounted read-only into the container.

After editing nginx.conf, reload without downtime:

```bash
docker exec trichome-nginx nginx -t          # validate config
docker exec trichome-nginx nginx -s reload   # hot reload
```

### Rate Limiting

Nginx enforces two rate limit zones:

| Zone      | Limit  | Burst | Applied to                     |
|-----------|--------|-------|--------------------------------|
| `api`     | 30 r/s | 60    | `/api/v1/*` REST endpoints     |
| `inference` | 5 r/s | 10  | (reserved for heavy GPU routes)|

Adjust in `nginx.conf` → `limit_req_zone` directives.

### Large Image Uploads

`client_max_body_size 512M` is set globally and on annotation paths. This supports
uncompressed 16-bit TIFF microscopy images up to 512 MB.

---

## TLS / HTTPS (Future)

The nginx config contains a commented HTTPS stub. To enable:

1. Obtain certs via certbot (requires public DNS pointing to host):
   ```bash
   certbot certonly --standalone -d your-domain.com \
     --pre-hook "docker stop trichome-nginx" \
     --post-hook "docker start trichome-nginx"
   ```

2. Mount certs into nginx container:
   ```yaml
   volumes:
     - /etc/letsencrypt/live/your-domain.com:/etc/nginx/certs:ro
   ```

3. Uncomment the `server { listen 443 ssl http2; ... }` block in `nginx.conf`.

4. Forward port **443** in addition to (or instead of) 3001 on the router.

> Until TLS is configured, restrict `/mlflow/` and `/annotation/` to LAN access
> only (add `allow 192.168.0.0/16; deny all;` inside those location blocks).

---

## Security Hardening (Production Checklist)

- [ ] Enable TLS (certbot + Let's Encrypt)
- [ ] Add HTTP Basic Auth in front of `/mlflow/` and `/cvat/` nginx locations
- [ ] Set `LABEL_STUDIO_SECRET_KEY` to a strong random value
- [ ] Set `LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true`
- [ ] Set `POSTGRES_PASSWORD` to a strong random value
- [ ] Change `CVAT_PASSWORD` from `admin`
- [ ] Add `X-Frame-Options`, `HSTS` headers to nginx (already present for same-origin)
- [ ] Restrict `CUDA_VISIBLE_DEVICES` in production — inference only
- [ ] Enable nginx access log → ship to central log store
- [ ] Set up UFW rules: allow only 22 (SSH), 3001, 443

---

*Last updated: 2026-05-25*
