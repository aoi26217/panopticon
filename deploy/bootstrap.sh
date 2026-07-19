#!/usr/bin/env bash
# ============================================================================
# Project Panopticon — Production Bootstrap
# deploy/bootstrap.sh
#
# One command from bare RunPod host to launch-ready:
#   1. system dependencies (curl, jq, python3-pip, openssl)
#   2. NVIDIA Container Runtime hook mapped into Docker
#   3. model weights pre-pulled into /workspace/models (HF_HOME cache — the
#      same cache docker-compose mounts, so sglang/TEI cold-boot in seconds,
#      not tens of minutes)
#   4. /workspace durable directory tree
#   5. .env with randomized secrets (0600, never overwritten without --force)
#
# Idempotent by design: every step detects done-ness and resumes. Model pulls
# retry with backoff and resume partial downloads (HF cache semantics).
#
# Flags:  --skip-system   --skip-models   --force-env
# ============================================================================
set -Eeuo pipefail

WORKSPACE="${PANOPTICON_WORKSPACE:-/workspace}"
MODELS_DIR="${WORKSPACE}/models"
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
MODELS=("Qwen/Qwen3-8B" "BAAI/bge-m3")
SKIP_SYSTEM=0 SKIP_MODELS=0 FORCE_ENV=0

for arg in "$@"; do
  case "$arg" in
    --skip-system) SKIP_SYSTEM=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --force-env)   FORCE_ENV=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap][warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bootstrap][fail]\033[0m %s\n' "$*" >&2; exit 1; }

retry() {  # retry <attempts> <sleep_base_s> <cmd...>
  local attempts="$1" base="$2" n=1; shift 2
  until "$@"; do
    if (( n >= attempts )); then return 1; fi
    local delay=$(( base * n ))
    warn "attempt ${n}/${attempts} failed: $* — retrying in ${delay}s"
    sleep "$delay"; n=$(( n + 1 ))
  done
}

# ---------------------------------------------------------------- 1. system
if (( ! SKIP_SYSTEM )); then
  log "system dependencies"
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    retry 3 5 apt-get update -qq || die "apt-get update failed"
    apt-get install -y -qq curl jq openssl python3-pip python3-venv \
      >/dev/null || die "apt-get install failed"
  else
    warn "apt-get not found — assuming deps preinstalled (non-Debian host)"
  fi
else
  log "system dependencies: SKIPPED (--skip-system)"
fi

# ------------------------------------------------- 2. NVIDIA runtime mapping
log "NVIDIA container runtime"
command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi missing — this is not a GPU host"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  | sed 's/^/[bootstrap]   GPU /'

if docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
  log "docker↔nvidia runtime hook: already mapped"
elif command -v nvidia-ctk >/dev/null 2>&1; then
  log "mapping runtime via nvidia-ctk"
  nvidia-ctk runtime configure --runtime=docker \
    || die "nvidia-ctk configure failed"
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active docker \
      >/dev/null 2>&1; then
    systemctl restart docker || die "docker restart failed"
  else
    warn "restart the docker daemon manually to activate the nvidia runtime"
  fi
else
  die "nvidia runtime not mapped and nvidia-ctk unavailable — install \
nvidia-container-toolkit first"
fi

# --------------------------------------------------------- 3. model pre-pull
if (( ! SKIP_MODELS )); then
  log "model pre-pull → ${MODELS_DIR} (HF_HOME cache, resumable)"
  mkdir -p "${MODELS_DIR}"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    pip3 install --quiet --upgrade "huggingface_hub[cli]" \
      || die "huggingface_hub install failed"
  fi
  export HF_HOME="${MODELS_DIR}" HF_HUB_ENABLE_HF_TRANSFER=0
  for repo in "${MODELS[@]}"; do
    log "  pulling ${repo}"
    retry 4 20 huggingface-cli download "${repo}" \
      --exclude "*.pt" --exclude "*.onnx" --quiet \
      || die "download failed after retries: ${repo}"
  done
  log "model cache ready: $(du -sh "${MODELS_DIR}" 2>/dev/null | cut -f1)"
else
  log "model pre-pull: SKIPPED (--skip-models)"
fi

# ------------------------------------------------------ 4. durable dir tree
log "durable directory tree under ${WORKSPACE}"
mkdir -p "${WORKSPACE}"/{models,qdrant,deadletter,state,prometheus,grafana} \
         "${WORKSPACE}"/neo4j/{data,logs}

# ------------------------------------------------------------ 5. .env secrets
if [[ -f "${ENV_FILE}" && ${FORCE_ENV} -eq 0 ]]; then
  log ".env exists — keeping current secrets (use --force-env to regenerate)"
else
  log "generating ${ENV_FILE} with randomized secrets"
  NEO4J_PW="$(openssl rand -hex 24)"
  LK_KEY="lk_$(openssl rand -hex 8)"
  LK_SECRET="$(openssl rand -hex 32)"
  GRAFANA_PW="$(openssl rand -hex 16)"
  umask 177
  cat > "${ENV_FILE}" <<EOF
# Project Panopticon — generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by bootstrap.sh
# chmod 600; never commit this file.
NEO4J_PASSWORD=${NEO4J_PW}
LIVEKIT_API_KEY=${LK_KEY}
LIVEKIT_API_SECRET=${LK_SECRET}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_PW}
GRAFANA_ANONYMOUS=false
EOF
  umask 022
  log ".env written (mode 600)"
fi

log "──────────────────────────────────────────────────────────────"
log "bootstrap complete. Launch sequence:"
log "  cd $(dirname "${ENV_FILE}")"
log "  docker compose -f docker-compose-runpod.yml --profile obs up -d"
log "  # after health checks clear, profile the silicon BEFORE trusting AIMD:"
log "  python3 ../profile_silicon.py --sglang-url http://localhost:8000"
log "──────────────────────────────────────────────────────────────"
