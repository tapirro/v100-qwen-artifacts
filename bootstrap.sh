#!/usr/bin/env bash
# bootstrap.sh — runs on a fresh vast.ai V100 32GB instance to produce a
# production Qwen3.6-35B-A3B llama-server at 6 × 256K concurrent slots.
#
# Target: ready in 10-15 minutes from empty container.
#
# Prereqs: container image must have libcuda.so + cublas (use nvidia/cuda:12.2.2-runtime-ubuntu22.04)
set -euo pipefail

echo "=== bootstrap start: $(date -u +%FT%TZ) ==="

LLAMA_BIN_URL="${LLAMA_BIN_URL:-https://github.com/tapirro/v100-qwen-artifacts/releases/download/v1/llama-server-v100-sm70.tar.gz}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"

TOTAL_CTX="${TOTAL_CTX:-1536000}"    # 6 × 256K
NUM_PARALLEL="${NUM_PARALLEL:-6}"
KV_QUANT="${KV_QUANT:-q4_0}"

# -------- 1. System deps (minimal) --------
echo "[1/5] apt install deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends \
  curl ca-certificates tmux libgomp1 python3 python3-pip >/dev/null
update-ca-certificates >/dev/null 2>&1 || true

# -------- 2. Fetch pre-built llama-server + libs --------
echo "[2/5] fetching pre-built llama-server tarball ($LLAMA_BIN_URL)"
mkdir -p /workspace/llama-bin
cd /workspace/llama-bin
curl -fsSL -o llama-server-v100-sm70.tar.gz "$LLAMA_BIN_URL"
tar xzf llama-server-v100-sm70.tar.gz
rm llama-server-v100-sm70.tar.gz
chmod +x /workspace/llama-bin/llama-server /workspace/llama-bin/llama-cli /workspace/llama-bin/llama-bench 2>/dev/null || true
ls -lh /workspace/llama-bin/

# -------- 3. hf CLI + download model --------
echo "[3/5] install hf-transfer + download model (~21GB)"
pip install -q --no-input "huggingface_hub[cli]" hf-transfer 2>&1 | tail -1 || true
export PATH="/usr/local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/models
cd /workspace/models
# Try hf (new) then huggingface-cli (legacy)
if command -v hf >/dev/null 2>&1; then HFCMD=hf; else HFCMD=huggingface-cli; fi
echo "Using $HFCMD"
time $HFCMD download "$MODEL_REPO" "$MODEL_FILE" --local-dir /workspace/models/q4kxl

# -------- 4. Launch llama-server in tmux --------
echo "[4/5] launching llama-server"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/workspace/llama-bin:${LD_LIBRARY_PATH:-}"

# Kill any prior server/session
tmux kill-session -t qwen 2>/dev/null || true
pkill -9 -f llama-server 2>/dev/null || true
sleep 2

tmux new-session -d -s qwen "LD_LIBRARY_PATH='$LD_LIBRARY_PATH' /workspace/llama-bin/llama-server \
  -m /workspace/models/q4kxl/$MODEL_FILE \
  -c $TOTAL_CTX -np $NUM_PARALLEL -ngl 99 -fa on \
  --cache-type-k $KV_QUANT --cache-type-v $KV_QUANT \
  --fit off \
  --slot-prompt-similarity 0.05 \
  --cache-ram 16384 \
  --host 0.0.0.0 --port 8080 --alias qwen3.6-35b-a3b \
  > /workspace/llama-server.log 2>&1"

# -------- 5. Wait for health --------
echo "[5/5] waiting for health"
READY=0
for i in $(seq 1 180); do
  if curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 1
done

if [[ $READY -ne 1 ]]; then
  echo "FAIL: server did not become healthy in 180s"
  tail -40 /workspace/llama-server.log
  exit 2
fi

# Report metadata
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader | head -1
echo "health: $(curl -sf http://127.0.0.1:8080/health)"
SLOTS=$(curl -sf http://127.0.0.1:8080/slots 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
echo "slots: $SLOTS"
echo "=== bootstrap complete: $(date -u +%FT%TZ) ==="
echo ""
echo "Endpoint: http://127.0.0.1:8080"
echo "tmux session: qwen (attach: tmux attach -t qwen)"
echo "Server log: /workspace/llama-server.log"
