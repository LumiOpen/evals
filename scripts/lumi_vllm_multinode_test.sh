#!/usr/bin/env bash
set -euo pipefail

export IMG="/scratch/project_462000353/danizaut/containers/vllm_v10.1.1.sif.bak"
export PRJ="/scratch/project_462000353"   # /project inside container
export SCR="$PWD"                          # /workspace inside container
export ACC="project_462000353"
export PART="dev-g"

# ---- multinode topology knobs ----
export N_NODES=${N_NODES:-2}
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export TP=${TP:-2}
export PP=${PP:-$N_NODES}
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
export JOB_TIME=${JOB_TIME:-02:00:00}
export RAY_PORT=${RAY_PORT:-43007}
export RAY_IFACE=${RAY_IFACE:-hsn0}
export OUT_BASENAME="${OUT_BASENAME:-multinode_smoketest}"

TOTAL_GPUS=$(( N_NODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS == 0 )); then
  echo "TOTAL_GPUS resolved to zero" >&2
  exit 2
fi
if (( TP > GPUS_PER_NODE )); then
  echo "TP ($TP) must not exceed GPUS_PER_NODE ($GPUS_PER_NODE)" >&2
  exit 2
fi

srun -A "$ACC" -p "$PART" -N "$N_NODES" --ntasks="$N_NODES" --ntasks-per-node=1 \
     --cpus-per-task=32 --gpus-per-task="$GPUS_PER_NODE" \
     --time="$JOB_TIME" --gpu-bind=closest \
  singularity exec --rocm --cleanenv \
    --bind "$SCR":/workspace \
    --bind "$PRJ":/project \
    --bind /usr/share/libdrm:/usr/share/libdrm \
    --env MODEL_ID="$MODEL_ID" \
    --env N_NODES="$N_NODES" \
    --env GPUS_PER_NODE="$GPUS_PER_NODE" \
    --env TOTAL_GPUS="$TOTAL_GPUS" \
    --env TP="$TP" \
    --env PP="$PP" \
    --env RAY_PORT="$RAY_PORT" \
    --env RAY_IFACE="$RAY_IFACE" \
    --env OUT_BASENAME="$OUT_BASENAME" \
    "$IMG" bash -lc '
set -euo pipefail
umask 002

RANK=${SLURM_PROCID:-0}
WORLD_SIZE=${SLURM_NTASKS:-1}
RUN_ROOT=/workspace
TOOLS_DIR="$RUN_ROOT/tools"
SENTINEL_DIR="$TOOLS_DIR/.sentinels"
OUTDIR="$RUN_ROOT/inference/${OUT_BASENAME}"
export OUTDIR

mkdir -p "$TOOLS_DIR" "$SENTINEL_DIR" "$OUTDIR"

# ---- env & caches ----
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/project/hf-cache
export HUGGINGFACE_HUB_CACHE=/project/hf-cache/hub
export TRANSFORMERS_CACHE=/project/hf-cache/models
export HF_DATASETS_CACHE=/project/hf-cache/datasets
export XDG_CACHE_HOME=/project/hf-cache/xdg
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
export TORCHINDUCTOR_CACHE_DIR=/project/hf-cache/torchinductor
export VLLM_COMPILER_CACHE_DIR=/project/hf-cache/vllm-compile
export PATH="$HOME/.local/bin:/opt/miniconda3/envs/pytorch/bin:/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/bin:/bin:$PATH"
export VLLM_TARGET_DEVICE=rocm
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_ARCHITECTURES=gfx90a
export VLLM_USE_RAY=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
export RCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
unset HIP_VISIBLE_DEVICES

mkdir -p /project/hf-cache/{hub,models,datasets,torchinductor,xdg,vllm-compile} \
         /dev/shm/torch_ext "$HOME/.aiter/jit/build" "$HOME/.aiter/jit/install"

if command -v /opt/rocm/llvm/bin/clang++ >/dev/null 2>&1; then
  export CC=/opt/rocm/llvm/bin/clang
  export CXX=/opt/rocm/llvm/bin/clang++
else
  export CC=/opt/rocm/bin/hipcc
  export CXX=/opt/rocm/bin/hipcc
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "[fatal] ray CLI not found in PATH=$PATH" >&2
  exit 3
fi

sentinel_put() { echo "$1" > "$SENTINEL_DIR/$2"; }
sentinel_wait() { local file="$SENTINEL_DIR/$1"; while [[ ! -s "$file" ]]; do sleep 2; done; cat "$file"; }

# Prefer the HSN interface but fall back gracefully
detect_ip() {
  local iface="$1"
  local ip=""
  if [[ -n "$iface" ]] && command -v ip >/dev/null 2>&1; then
    ip=$(ip -o -4 addr show dev "$iface" 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -n1)
  fi
  if [[ -z "$ip" ]] && command -v hostname >/dev/null 2>&1; then
    ip=$(hostname -I | tr " " "\n" | grep -m1 -E "^10\\.|^172\\.16|^172\\.3|^192\\.168" || true)
  fi
  if [[ -z "$ip" ]] && command -v hostname >/dev/null 2>&1; then
    ip=$(hostname -I | awk "{print \$1}")
  fi
  if [[ -z "$ip" ]] && [[ -f /etc/hostname ]] && [[ -f /etc/hosts ]]; then
    local host_name
    host_name=$(cat /etc/hostname 2>/dev/null || echo '')
    if [[ -n "$host_name" ]]; then
      ip=$(grep -m1 "$host_name" /etc/hosts | awk '{print $1}' || true)
    fi
  fi
  if [[ -z "$ip" ]]; then
    ip=$(python - <<'PY'
import socket
try:
    host = socket.gethostname()
    addr = socket.gethostbyname(host)
    # LUMI hostnames resolve to internal IPs; print only IPv4
    if addr:
        print(addr)
except Exception:
    pass
PY
)
  fi
  printf '%s\n' "$ip"
}

HOST_IP=$(detect_ip "${RAY_IFACE:-hsn0}")
if [[ -z "$HOST_IP" ]]; then
  echo "Failed to determine IP on rank $RANK" >&2
  exit 2
fi

if [[ "$RANK" -eq 0 ]]; then
  HEAD_IP="$HOST_IP"
  sentinel_put "$HEAD_IP" head.ip
else
  HEAD_IP=$(sentinel_wait head.ip)
fi

if [[ "$RANK" -eq 0 ]]; then
  python <<'PY'
import torch
print("[sanity] torch", torch.__version__, "HIP", getattr(torch.version, "hip", None))
print("[sanity] visible gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
PY
  python <<'PY'
from huggingface_hub import snapshot_download
import os
model_id = os.environ["MODEL_ID"]
local = f"/project/hf-cache/models/{model_id.replace("/", "-") }"
path = snapshot_download(repo_id=model_id,
                         local_dir=local,
                         local_dir_use_symlinks=False,
                         allow_patterns=["*.safetensors","*.json","tokenizer.*","*vocab*","*.model"])
print("[prefetch] cached ->", path)
open("/workspace/tools/.sentinels/prefetch.done","w").write(path)
PY
else
  sentinel_wait prefetch.done >/dev/null
fi

export HF_HUB_OFFLINE=1
export HF_HUB_READ_FROM_CACHE_ONLY=1

start_ray_head() {
  mkdir -p /dev/shm/ray
  ray stop --force >/dev/null 2>&1 || true
  ray start --head \
    --node-ip-address="$HOST_IP" \
    --port="$RAY_PORT" \
    --disable-usage-stats \
    --num-gpus="$GPUS_PER_NODE" \
    --temp-dir=/dev/shm/ray \
    --block &
}

start_ray_worker() {
  mkdir -p /dev/shm/ray
  ray stop --force >/dev/null 2>&1 || true
  ray start \
    --address="$HEAD_IP:$RAY_PORT" \
    --node-ip-address="$HOST_IP" \
    --disable-usage-stats \
    --num-gpus="$GPUS_PER_NODE" \
    --temp-dir=/dev/shm/ray \
    --block &
}

if [[ "$RANK" -eq 0 ]]; then
  start_ray_head
  sentinel_put "$HOST_IP" ray.head.ready
else
  sentinel_wait ray.head.ready >/dev/null
  start_ray_worker
fi

wait_for_cluster() {
python <<'PY'
import os, time
import ray
head_ip = os.environ['HEAD_IP']
port = os.environ['RAY_PORT']
address = f"ray://{head_ip}:10001"
for attempt in range(40):
    try:
        ray.init(address="auto", namespace="vllm", ignore_reinit_error=True)
        ray.shutdown()
        break
    except Exception as exc:
        print(f"[ray-wait] attempt {attempt+1}: {exc}")
        time.sleep(6)
else:
    raise SystemExit("Ray cluster did not become ready in time")
PY
}

if [[ "$RANK" -eq 0 ]]; then
  export HEAD_IP
  wait_for_cluster
  python <<'PY'
import os
from vllm import LLM, SamplingParams

model_id = os.environ['MODEL_ID']
model_local = f"/project/hf-cache/models/{model_id.replace("/", "-") }"
out_dir = os.environ['OUTDIR']

llm = LLM(
    model=model_local,
    tensor_parallel_size=int(os.environ['TP']),
    pipeline_parallel_size=int(os.environ['PP']),
    distributed_executor_backend='ray',
    worker_use_ray=True,
)

prompt = "You are running on the LUMI supercomputer. Write a single sentence announcing that multi-node vLLM inference works."
params = SamplingParams(temperature=0.7, max_tokens=80)
outputs = llm.generate([prompt], params)
out = outputs[0].outputs[0].text.strip()
print("[inference] prompt:", prompt)
print("[inference] completion:")
print(out)
with open(os.path.join(out_dir, "response_rank0.txt"), "w") as fh:
    fh.write(out + "\n")
PY
  sentinel_put done inference.done
  ray stop --force || true
else
  sentinel_wait inference.done >/dev/null
  ray stop --force || true
fi

'
