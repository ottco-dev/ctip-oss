#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ctip.sh — CTIP Platform Manager
#
# Usage:
#   ./ctip.sh start              Start backend + frontend
#   ./ctip.sh stop               Graceful stop (SIGTERM)
#   ./ctip.sh kill               Force stop (SIGKILL)
#   ./ctip.sh restart            kill → clean → start
#   ./ctip.sh status             Show port / process / GPU status
#   ./ctip.sh logs [svc]         Tail logs  (backend | frontend | mlflow | all)
#   ./ctip.sh clean              Remove build artefacts (.next)
#   ./ctip.sh docker [up|down]   Label Studio + annotation containers
# ─────────────────────────────────────────────────────────────────────────────

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv/bin/activate"
LOGDIR="$REPO/logs"

BACKEND_PORT=8000
FRONTEND_PORT=3000
MLFLOW_PORT=3004
LS_PORT=3005

BACKEND_LOG="$LOGDIR/backend.log"
FRONTEND_LOG="$LOGDIR/frontend.log"
MLFLOW_LOG="$LOGDIR/mlflow.log"

mkdir -p "$LOGDIR"

# ── colours ──────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
C='\033[0;36m'; B='\033[1;34m'; DIM='\033[2m'; NC='\033[0m'
BOLD='\033[1m'

ok()  { printf "  ${G}✓${NC}  %s\n" "$*"; }
err() { printf "  ${R}✗${NC}  %s\n" "$*"; }
inf() { printf "  ${C}→${NC}  %s\n" "$*"; }
wrn() { printf "  ${Y}!${NC}  %s\n" "$*"; }
hdr() { printf "\n${B}${BOLD}  %s${NC}\n  %s\n" "$*" "$(printf '─%.0s' {1..54})"; }

# ── helpers ───────────────────────────────────────────────────────────────────

port_pid()  { ss -tlnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | head -1; }
port_used() { ss -tlnp 2>/dev/null | grep -q ":$1 "; }

wait_port() {
    local port=$1 name=$2 max=${3:-20}
    printf "  ${DIM}Waiting for %s on :%s${NC}" "$name" "$port"
    for ((i=1; i<=max; i++)); do
        if port_used "$port"; then printf "\r"; ok "$name on :$port"; return 0; fi
        printf "."
        sleep 1
    done
    printf "\r"; err "$name failed to start on :$port (${max}s timeout) — see $LOGDIR/"
    return 1
}

kill_port() {
    local port=$1 sig=${2:--TERM}
    local pid; pid="$(port_pid "$port")"
    if [[ -n "$pid" ]]; then
        kill "$sig" "$pid" 2>/dev/null && return 0
    fi
    return 1
}

clean_next() {
    # .next may contain root-owned files written by Docker
    if [[ -d "$REPO/frontend/.next" ]]; then
        local root_files
        root_files=$(find "$REPO/frontend/.next" -user root 2>/dev/null | wc -l)
        if (( root_files > 0 )); then
            inf "Clearing $root_files root-owned .next files via Docker container…"
            sg docker -c "docker exec trichome-frontend sh -c 'rm -rf /app/.next/static /app/.next/server /app/.next/cache/webpack'" 2>/dev/null || true
        fi
        rm -rf "$REPO/frontend/.next" 2>/dev/null && inf ".next cleared" || true
    fi
}

# ── stop / kill ───────────────────────────────────────────────────────────────

do_stop() {
    local sig=${1:--TERM}
    local sig_name="SIGTERM"; [[ "$sig" == "-KILL" ]] && sig_name="SIGKILL"
    hdr "Stopping CTIP services  ($sig_name)"

    # Backend
    if kill_port $BACKEND_PORT "$sig"; then
        ok "Backend stopped  (:$BACKEND_PORT)"
    else
        pkill "$sig" -f "uvicorn backend.main" 2>/dev/null && ok "Backend stopped" || inf "Backend was not running"
    fi

    # Frontend (Next.js — multiple related processes)
    if kill_port $FRONTEND_PORT "$sig"; then
        ok "Frontend stopped  (:$FRONTEND_PORT)"
    fi
    pkill "$sig" -f "next dev"    2>/dev/null || true
    pkill "$sig" -f "next-server" 2>/dev/null || true
    pkill "$sig" -f "next build"  2>/dev/null || true
    # If processes survive TERM, give them 2 s then KILL
    if [[ "$sig" == "-TERM" ]]; then
        sleep 2
        pkill -KILL -f "next dev"    2>/dev/null || true
        pkill -KILL -f "next-server" 2>/dev/null || true
    fi
    ok "Frontend stopped"

    # MLflow (optional — don't error if absent)
    pkill "$sig" -f "mlflow"  2>/dev/null && ok "MLflow stopped" || true
    pkill "$sig" -f "gunicorn.*mlflow" 2>/dev/null || true

    # Wait for ports to free
    sleep 1
    local dirty=0
    for port in $BACKEND_PORT $FRONTEND_PORT; do
        if port_used "$port"; then
            wrn "Port :$port still occupied — forcing kill"
            kill_port "$port" -KILL 2>/dev/null || true
            dirty=1
        fi
    done
    [[ $dirty -eq 1 ]] && sleep 1

    echo ""
}

# ── clean ─────────────────────────────────────────────────────────────────────

cmd_clean() {
    hdr "Cleaning build artefacts"
    clean_next
    find "$LOGDIR" -name "*.log" -size 0 -delete 2>/dev/null || true
    ok "Clean complete"
    echo ""
}

# ── start ─────────────────────────────────────────────────────────────────────

cmd_start() {
    hdr "Starting CTIP services"
    cd "$REPO"

    # Safety: abort if ports are already occupied (offer to kill)
    local blocked=0
    for port in $BACKEND_PORT $FRONTEND_PORT; do
        if port_used "$port"; then
            local pid; pid="$(port_pid "$port")"
            wrn "Port :$port is already in use (PID $pid)"
            blocked=1
        fi
    done
    if (( blocked )); then
        printf "\n  Kill existing processes and continue? [y/N] "
        read -r ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            do_stop -KILL
        else
            err "Aborted."
            exit 1
        fi
    fi

    # ── Backend ──────────────────────────────────────────────────────────────
    inf "Starting backend (FastAPI :$BACKEND_PORT)…"
    # Launch via sg docker so the process inherits docker group membership.
    # Required for containers API (docker ps / docker compose) to work without sudo.
    # shellcheck disable=SC1090
    source "$VENV"
    PYTHON="$REPO/.venv/bin/python"
    if id -nG "$USER" 2>/dev/null | grep -qw docker && ! id -G 2>/dev/null | grep -qw "$(getent group docker | cut -d: -f3)"; then
        # User is in docker group but current session doesn't have it — use sg
        sg docker -c "$PYTHON -m uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload --reload-dir $REPO/backend --reload-dir $REPO/training --reload-dir $REPO/shared >> $BACKEND_LOG 2>&1 &"
    else
        $PYTHON -m uvicorn backend.main:app \
            --host 0.0.0.0 \
            --port "$BACKEND_PORT" \
            --reload \
            --reload-dir "$REPO/backend" \
            --reload-dir "$REPO/training" \
            --reload-dir "$REPO/shared" \
            >> "$BACKEND_LOG" 2>&1 &
    fi
    wait_port $BACKEND_PORT "Backend" 20

    # ── Frontend ─────────────────────────────────────────────────────────────
    clean_next
    inf "Starting frontend (Next.js :$FRONTEND_PORT)…"
    cd "$REPO/frontend"
    npm run dev -- --port "$FRONTEND_PORT" >> "$FRONTEND_LOG" 2>&1 &
    cd "$REPO"
    wait_port $FRONTEND_PORT "Frontend" 30

    # ── MLflow (optional) ────────────────────────────────────────────────────
    if ! port_used $MLFLOW_PORT; then
        inf "Starting MLflow (:$MLFLOW_PORT)…"
        mkdir -p "$REPO/mlruns"
        mlflow ui \
            --host 0.0.0.0 \
            --port "$MLFLOW_PORT" \
            --backend-store-uri "$REPO/mlruns" \
            >> "$MLFLOW_LOG" 2>&1 &
        wait_port $MLFLOW_PORT "MLflow" 10 || wrn "MLflow did not start (non-fatal)"
    else
        ok "MLflow already on :$MLFLOW_PORT"
    fi

    _print_urls
}

# ── status ────────────────────────────────────────────────────────────────────

cmd_status() {
    hdr "CTIP service status"

    svc_line() {
        local port=$1 name=$2 url=$3
        if port_used "$port"; then
            local pid; pid="$(port_pid "$port")"
            printf "  ${G}●${NC}  %-22s :%-5s  ${DIM}pid %-7s${NC}  %s\n" "$name" "$port" "$pid" "$url"
        else
            printf "  ${R}●${NC}  %-22s :%-5s  ${DIM}not running${NC}\n" "$name" "$port"
        fi
    }

    svc_line $BACKEND_PORT  "Backend (FastAPI)"   "http://localhost:$BACKEND_PORT/docs"
    svc_line $FRONTEND_PORT "Frontend (Next.js)"  "http://localhost:$FRONTEND_PORT"
    svc_line $MLFLOW_PORT   "MLflow"              "http://localhost:$MLFLOW_PORT"
    svc_line $LS_PORT       "Label Studio"        "http://localhost:$LS_PORT"

    # GPU
    echo ""
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name vram_used vram_total temp
        gpu_name=$(nvidia-smi --query-gpu=name          --format=csv,noheader 2>/dev/null | head -1)
        vram_used=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null | head -1)
        vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        temp=$(nvidia-smi --query-gpu=temperature.gpu  --format=csv,noheader,nounits 2>/dev/null | head -1)
        printf "  ${G}▸${NC}  GPU  %-22s  %s/%s MiB  %s°C\n" \
            "$gpu_name" "$vram_used" "$vram_total" "$temp"
    else
        wrn "nvidia-smi not found"
    fi

    # Log sizes
    echo ""
    for f in "$BACKEND_LOG" "$FRONTEND_LOG" "$MLFLOW_LOG"; do
        [[ -f "$f" ]] && printf "  ${DIM}%-40s  %s${NC}\n" "$(basename "$f")" "$(wc -l < "$f") lines"
    done
    echo ""
}

# ── logs ──────────────────────────────────────────────────────────────────────

cmd_logs() {
    local svc="${1:-all}"
    case "$svc" in
        backend|be)   tail -f "$BACKEND_LOG" ;;
        frontend|fe)  tail -f "$FRONTEND_LOG" ;;
        mlflow|ml)    tail -f "$MLFLOW_LOG" ;;
        all)
            hdr "Streaming all logs  (Ctrl-C to stop)"
            tail -f "$BACKEND_LOG" "$FRONTEND_LOG" "$MLFLOW_LOG" 2>/dev/null ;;
        *)
            err "Unknown service '$svc'. Use: backend | frontend | mlflow | all"
            exit 1 ;;
    esac
}

# ── docker helpers ────────────────────────────────────────────────────────────

cmd_docker() {
    local action="${1:-up}"
    local compose="$REPO/docker/docker-compose.annotation.yml"
    cd "$REPO/docker"
    case "$action" in
        up)
            hdr "Starting annotation containers"
            sg docker -c "docker compose -f docker-compose.annotation.yml up -d" 2>&1
            echo ""
            ok "Label Studio  →  http://localhost:$LS_PORT" ;;
        down)
            hdr "Stopping annotation containers"
            sg docker -c "docker compose -f docker-compose.annotation.yml down" 2>&1 ;;
        restart)
            sg docker -c "docker compose -f docker-compose.annotation.yml restart" 2>&1 ;;
        status)
            sg docker -c "docker compose -f docker-compose.annotation.yml ps" 2>&1 ;;
        logs)
            sg docker -c "docker compose -f docker-compose.annotation.yml logs -f --tail=100" 2>&1 ;;
        *)
            err "Unknown docker action. Use: up | down | restart | status | logs" ;;
    esac
    cd "$REPO"
}

# ── URLs ──────────────────────────────────────────────────────────────────────

_print_urls() {
    echo ""
    printf "  ${G}${BOLD}CTIP ready.${NC}\n\n"
    printf "  %-30s %s\n"  "Frontend:"       "http://localhost:$FRONTEND_PORT"
    printf "  %-30s %s\n"  "API (Swagger):"  "http://localhost:$BACKEND_PORT/docs"
    printf "  %-30s %s\n"  "MLflow:"         "http://localhost:$MLFLOW_PORT"
    printf "  %-30s %s\n"  "Label Studio:"   "http://localhost:$LS_PORT"
    echo ""
    printf "  ${DIM}Logs:${NC}  ./ctip.sh logs [backend|frontend|mlflow|all]\n"
    echo ""
}

# ── dispatch ──────────────────────────────────────────────────────────────────

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
    start)         cmd_start ;;
    stop)          do_stop -TERM; echo "" ;;
    kill|force)    do_stop -KILL; echo "" ;;
    restart)       do_stop -KILL; sleep 1; cmd_clean; cmd_start ;;
    status|st)     cmd_status ;;
    logs|log)      cmd_logs "$@" ;;
    clean)         cmd_clean ;;
    docker|dc)     cmd_docker "$@" ;;
    urls)          _print_urls ;;
    help|--help|-h)
        echo ""
        printf "  ${BOLD}ctip.sh — CTIP Platform Manager${NC}\n\n"
        printf "  ${C}Commands:${NC}\n"
        printf "    ${BOLD}start${NC}              Start backend + frontend + MLflow\n"
        printf "    ${BOLD}stop${NC}               Graceful stop (SIGTERM)\n"
        printf "    ${BOLD}kill${NC}               Force stop (SIGKILL)\n"
        printf "    ${BOLD}restart${NC}            Force kill → clean → start fresh\n"
        printf "    ${BOLD}status${NC}  (st)       Show ports, PIDs, GPU stats\n"
        printf "    ${BOLD}logs${NC}  [svc]        Tail logs (backend|frontend|mlflow|all)\n"
        printf "    ${BOLD}clean${NC}              Remove .next build cache\n"
        printf "    ${BOLD}docker${NC}  [action]   Annotation containers (up|down|restart|status|logs)\n"
        printf "    ${BOLD}urls${NC}               Print service URLs\n"
        echo ""
        printf "  ${C}Examples:${NC}\n"
        printf "    ./ctip.sh restart\n"
        printf "    ./ctip.sh status\n"
        printf "    ./ctip.sh logs backend\n"
        printf "    ./ctip.sh docker up\n"
        echo ""
        ;;
    *)
        err "Unknown command: $CMD"
        printf "  Run  ${BOLD}./ctip.sh help${NC}  for usage.\n\n"
        exit 1 ;;
esac
