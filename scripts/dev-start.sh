#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/dev-start.sh — Start all CTIP services for local development
#
# Services started:
#   :8000  Backend   (FastAPI/uvicorn, hot reload)
#   :3000  Frontend  (Next.js dev server)
#   :3001  nginx     (reverse proxy — /api/v1/ → :8000, / → :3000)
#   :3004  MLflow    (experiment tracking UI)
#
# Docker services (Label Studio :3005, CVAT :3006) require:
#   cd docker && docker compose --profile annotation up -d
#
# Usage:
#   ./scripts/dev-start.sh          # start all
#   ./scripts/dev-start.sh stop     # kill all managed processes
#   ./scripts/dev-start.sh status   # show what's running
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO/logs"
PIDS="$REPO/nginx-local/nginx.pid"
NGINX_CONF="$REPO/nginx-local/nginx.conf"
VENV="$REPO/.venv/bin/activate"

mkdir -p "$LOGS"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "  ${RED}✗${NC} $*"; }
hdr()  { echo -e "\n${CYAN}── $* ──────────────────────────────────────${NC}"; }

port_in_use() { ss -tlnp 2>/dev/null | grep -q ":$1 "; }

wait_port() {
    local port=$1 name=$2 timeout=${3:-15}
    for i in $(seq 1 $timeout); do
        if port_in_use "$port"; then ok "$name ready on :$port"; return 0; fi
        sleep 1
    done
    err "$name did not start on :$port within ${timeout}s — check $LOGS/"
    return 1
}

# ── stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
    hdr "Stopping CTIP dev services"

    # nginx — signal master via pidfile
    if [ -f "$PIDS" ] && kill -0 "$(cat "$PIDS")" 2>/dev/null; then
        kill "$(cat "$PIDS")" 2>/dev/null && ok "nginx stopped" || warn "nginx already stopped"
    else
        pkill -f "nginx.*nginx-local" 2>/dev/null && ok "nginx stopped" || true
    fi

    pkill -f "uvicorn backend.main:app"      2>/dev/null && ok "backend stopped"  || true
    pkill -f "next dev"                       2>/dev/null && ok "frontend stopped" || true
    pkill -f "next-server"                    2>/dev/null && true
    pkill -f "mlflow ui"                      2>/dev/null && ok "MLflow stopped"   || true
    pkill -f "gunicorn.*mlflow"              2>/dev/null || true

    echo ""
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
    hdr "CTIP dev service status"
    local all_ok=true

    check_svc() {
        local port=$1 name=$2
        if port_in_use "$port"; then
            ok "$name  (:$port)"
        else
            err "$name  (:$port)  NOT running"
            all_ok=false
        fi
    }

    check_svc 3001 "nginx (proxy)"
    check_svc 8000 "Backend (FastAPI)"
    check_svc 3000 "Frontend (Next.js)"
    check_svc 3004 "MLflow"

    echo ""
    if $all_ok; then
        echo -e "  ${GREEN}All core services running.${NC}"
        echo ""
        echo "  http://localhost:3001       ← main entry (via nginx)"
        echo "  http://localhost:3001/setup ← setup wizard"
        echo "  http://localhost:3004       ← MLflow"
        echo "  http://localhost:8000/docs  ← Swagger API docs"
    else
        warn "Some services are down. Run:  ./scripts/dev-start.sh"
    fi
    echo ""
}

# ── start ─────────────────────────────────────────────────────────────────────
cmd_start() {
    hdr "Starting CTIP dev services"
    cd "$REPO"

    # ── 1. nginx ──────────────────────────────────────────────────────────────
    if port_in_use 3001; then
        ok "nginx already running on :3001"
    else
        echo "  Starting nginx..."
        nginx -c "$NGINX_CONF"
        wait_port 3001 "nginx"
    fi

    # ── 2. Backend ────────────────────────────────────────────────────────────
    if port_in_use 8000; then
        ok "Backend already running on :8000"
    else
        echo "  Starting backend (FastAPI)..."
        # shellcheck disable=SC1090
        source "$VENV"
        uvicorn backend.main:app \
            --reload \
            --port 8000 \
            --host 0.0.0.0 \
            > "$LOGS/backend.log" 2>&1 &
        wait_port 8000 "Backend"
    fi

    # ── 3. Frontend ───────────────────────────────────────────────────────────
    if port_in_use 3000; then
        ok "Frontend already running on :3000"
    else
        echo "  Starting frontend (Next.js)..."
        cd "$REPO/frontend"
        npm run dev > "$LOGS/frontend.log" 2>&1 &
        cd "$REPO"
        wait_port 3000 "Frontend" 30
    fi

    # ── 4. MLflow ─────────────────────────────────────────────────────────────
    if port_in_use 3004; then
        ok "MLflow already running on :3004"
    else
        echo "  Starting MLflow..."
        # shellcheck disable=SC1090
        source "$VENV"
        mkdir -p "$REPO/mlruns"
        mlflow ui \
            --host 0.0.0.0 \
            --port 3004 \
            --backend-store-uri "$REPO/mlruns" \
            > "$LOGS/mlflow.log" 2>&1 &
        wait_port 3004 "MLflow" 10
    fi

    # ── Summary ───────────────────────────────────────────────────────────────
    echo ""
    echo -e "  ${GREEN}CTIP ready.${NC}"
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │  http://localhost:3001         Main UI (via nginx)  │"
    echo "  │  http://localhost:3001/setup   Setup Wizard         │"
    echo "  │  http://localhost:3004         MLflow               │"
    echo "  │  http://localhost:8000/docs    Swagger API Docs     │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo ""
    echo "  Logs:  tail -f logs/backend.log  |  logs/frontend.log  |  logs/mlflow.log"
    echo ""
    warn "Label Studio (:3005) and CVAT (:3006) require Docker:"
    echo "      cd docker && docker compose --profile annotation up -d"
    echo ""
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "${1:-start}" in
    start)  cmd_start  ;;
    stop)   cmd_stop   ;;
    status) cmd_status ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    *)
        echo "Usage: $0 [start|stop|restart|status]"
        exit 1
        ;;
esac
