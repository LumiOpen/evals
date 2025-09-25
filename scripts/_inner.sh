#!/usr/bin/env bash
set -euo pipefail
umask 002

if [[ -z ${SLURM_PROCID:-} ]]; then
  echo "[error] SLURM_PROCID missing inside container" >&2
  env | sort >&2
  exit 8
fi
if [[ -z ${SLURM_NTASKS:-} ]]; then
  echo "[error] SLURM_NTASKS missing inside container" >&2
  env | sort >&2
  exit 8
fi

RANK=${SLURM_PROCID}
WORLD_SIZE=${SLURM_NTASKS}
RUN_ROOT=/workspace
TOOLS_DIR="$RUN_ROOT/tools"
SENTINEL_DIR="$TOOLS_DIR/.sentinels"
OUTDIR="$RUN_ROOT/inference/${OUT_BASENAME}"
export OUTDIR

mkdir -p "$TOOLS_DIR" "$SENTINEL_DIR" "$OUTDIR"
rm -f "$SENTINEL_DIR"/* 2>/dev/null || true

# Log Ray env state before cleanup
echo "[ray-env] BEFORE cleanup: RAY_ADDRESS=${RAY_ADDRESS:-unset} RAY_GCS_SERVER_ADDRESS=${RAY_GCS_SERVER_ADDRESS:-unset}"
unset RAY_ADDRESS RAY_GCS_SERVER_ADDRESS RAY_GCS_ADDRESS RAY_HEAD_NODE_IP \
      RAY_HEAD_SERVICE_PORT RAY_HEAD_MANAGER_PORT RAY_BOOTSTRAP_ADDRESS \
      RAY_CLUSTER_LAUNCHER_ADDRESS RAY_REDIS_ADDRESS RAY_HEAD_ADDRESS
echo "[ray-env] AFTER cleanup: all Ray env vars unset"

# Fix for Ray v2.10.0+ compatibility with AMD GPUs and vLLM
# Prevents Ray from overriding ROCR_VISIBLE_DEVICES in workers
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
# Disable Ray Compiled DAG to avoid missing experimental channel module
export VLLM_USE_RAY_COMPILED_DAG=0
# Ensure vLLM doesn't override our Ray Compiled DAG setting
mkdir -p "$HOME/.config/vllm"
echo '["VLLM_USE_RAY_COMPILED_DAG"]' > "$HOME/.config/vllm/ray_non_carry_over_env_vars.json"
echo "[ray-env] Set RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1, disabled Ray Compiled DAG, and configured vLLM to not override it"
INFER_SCRIPT="$TOOLS_DIR/run_inference.py"

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
hip_mask="${ROCR_VISIBLE_DEVICES:-}"
if [[ -z $hip_mask && -n ${SLURM_STEP_GPUS:-} ]]; then
  hip_mask=$SLURM_STEP_GPUS
fi
if [[ -z $hip_mask && -n ${SLURM_JOB_GPUS:-} ]]; then
  hip_mask=$SLURM_JOB_GPUS
fi
if [[ -n $hip_mask ]]; then
  export ROCR_VISIBLE_DEVICES="$hip_mask"
  export HIP_VISIBLE_DEVICES="$hip_mask"
  export ROCM_VISIBLE_DEVICES="$hip_mask"
  export CUDA_VISIBLE_DEVICES="$hip_mask"
fi

host_name=$(cat /proc/sys/kernel/hostname 2>/dev/null || echo unknown)
echo "[env] rank=$RANK host=$host_name HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"

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

python -m pip -q install --user -U ninja || true

AITER_STAGE_SCRIPT="$TOOLS_DIR/stage_aiter.py"
cat > "$AITER_STAGE_SCRIPT" <<'PY'
import glob
import importlib
import os
import pathlib
import shutil
import subprocess
import time
import sys

home = os.path.expanduser("~")
jit_root = os.path.join(home, ".aiter", "jit")
build_root = os.path.join(jit_root, "build")
inst_root = os.path.join(jit_root, "install")
pkg_root = os.path.join(inst_root, "private_aiter")
pkg_jit = os.path.join(pkg_root, "jit")

os.makedirs(pkg_jit, exist_ok=True)
pathlib.Path(os.path.join(pkg_root, "__init__.py")).write_text("")
pathlib.Path(os.path.join(pkg_jit, "__init__.py")).write_text("")

try:
    import aiter  # noqa: F401
    from aiter.ops import enum  # noqa: F401
except Exception as exc:  # pragma: no cover
    print("[aiter] prewarm raised:", repr(exc))

pattern = os.path.join(build_root, "**", "module_aiter_enum*.so")
hits = []
for attempt in range(60):
    hits = glob.glob(pattern, recursive=True)
    if hits:
        break
    if attempt % 12 == 0:
        print("[aiter-stage] waiting for compiled module...")
    time.sleep(5)
else:
    raise SystemExit("[aiter-stage] no module_aiter_enum*.so found under " + build_root)

so_src = max(hits, key=os.path.getmtime)
dst = os.path.join(pkg_jit, "module_aiter_enum.so")
if os.path.islink(dst) or os.path.exists(dst):
    os.remove(dst)
try:
    os.symlink(so_src, dst)
    print("[aiter-stage] symlinked", dst, "->", so_src)
except OSError:
    shutil.copy2(so_src, dst)
    print("[aiter-stage] copied", so_src, "->", dst)

print("[aiter-stage] ldd:\n" + subprocess.check_output(["ldd", dst], text=True))

os.environ.setdefault("PYTHONPATH", inst_root)
if inst_root not in sys.path:
    sys.path.insert(0, inst_root)
import private_aiter.jit.module_aiter_enum  # noqa: F401,E402
print("[aiter-stage] import OK")
PY

python "$AITER_STAGE_SCRIPT"
export PYTHONPATH="$HOME/.aiter/jit/install:${PYTHONPATH-}"

sentinel_put() {
  local value="${1:-}"
  local token="${2:-}"
  if [[ -z "$token" ]]; then
    echo "[sentinel] token missing" >&2
    exit 4
  fi
  echo "$value" > "$SENTINEL_DIR/$token"
}

sentinel_wait() {
  local token="${1:-}"
  if [[ -z "$token" ]]; then
    echo "[sentinel] wait token missing" >&2
    exit 4
  fi
  local file="$SENTINEL_DIR/$token"
  while [[ ! -s "$file" ]]; do
    sleep 2
  done
  cat "$file"
}

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
export HOST_IP

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
  echo "[ray-head] starting on $HOST_IP:$RAY_PORT with $GPUS_PER_NODE GPUs"
  mkdir -p /dev/shm/ray
  ray stop --force >/dev/null 2>&1 || true
  # Clear any leftover Ray session state
  rm -rf /dev/shm/ray/* /tmp/ray* ~/.ray* 2>/dev/null || true
  echo "[ray-head] cleared session state, unsetting env vars"
  # Ensure Ray env variables are cleared
  unset RAY_ADDRESS RAY_GCS_SERVER_ADDRESS RAY_GCS_ADDRESS RAY_HEAD_NODE_IP \
        RAY_HEAD_SERVICE_PORT RAY_HEAD_MANAGER_PORT RAY_BOOTSTRAP_ADDRESS \
        RAY_CLUSTER_LAUNCHER_ADDRESS RAY_REDIS_ADDRESS RAY_HEAD_ADDRESS
  echo "[ray-head] launching Ray head with GPU mask: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  # Let Ray manage ROCR_VISIBLE_DEVICES automatically - only set it for Ray start
  env ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    ray start --head \
    --node-ip-address="$HOST_IP" \
    --port="$RAY_PORT" \
    --disable-usage-stats \
    --num-gpus="$GPUS_PER_NODE" \
    --temp-dir=/dev/shm/ray \
    --block &
}

start_ray_worker() {
  echo "[ray-worker] connecting to head at $HEAD_IP:$RAY_PORT from $HOST_IP with $GPUS_PER_NODE GPUs"
  mkdir -p /dev/shm/ray
  ray stop --force >/dev/null 2>&1 || true
  # Clear any leftover Ray session state
  rm -rf /dev/shm/ray/* /tmp/ray* ~/.ray* 2>/dev/null || true
  echo "[ray-worker] cleared session state, unsetting env vars"
  # Ensure Ray env variables are cleared
  unset RAY_ADDRESS RAY_GCS_SERVER_ADDRESS RAY_GCS_ADDRESS RAY_HEAD_NODE_IP \
        RAY_HEAD_SERVICE_PORT RAY_HEAD_MANAGER_PORT RAY_BOOTSTRAP_ADDRESS \
        RAY_CLUSTER_LAUNCHER_ADDRESS RAY_REDIS_ADDRESS RAY_HEAD_ADDRESS
  echo "[ray-worker] launching Ray worker with GPU mask: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  # Let Ray manage ROCR_VISIBLE_DEVICES automatically - only set it for Ray start
  env ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
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
  export HEAD_IP="$HOST_IP"
  export RAY_ADDRESS="$HOST_IP:$RAY_PORT"
  export VLLM_HOST_IP="$HOST_IP"
  echo "[vllm-ip] Head node: VLLM_HOST_IP=$VLLM_HOST_IP HOST_IP=$HOST_IP"
  sentinel_put "$HOST_IP" ray.head.ready
else
  HEAD_IP=$(sentinel_wait ray.head.ready)
  export HEAD_IP
  export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
  # Use HEAD_IP for worker VLLM_HOST_IP to avoid IP uniqueness issues
  export VLLM_HOST_IP="$HEAD_IP"
  echo "[vllm-ip] Worker node: VLLM_HOST_IP=$VLLM_HOST_IP HOST_IP=$HOST_IP HEAD_IP=$HEAD_IP"
  start_ray_worker
fi

wait_for_cluster() {
python <<'PY'
import os
import time
import ray

address = os.environ.get("RAY_ADDRESS")
if not address:
    head_ip = os.environ.get("HEAD_IP")
    port = os.environ.get("RAY_PORT", "43007")
    address = f"{head_ip}:{port}" if head_ip else None

print(f"[ray-wait] waiting for cluster at {address}")
for attempt in range(40):
    try:
        if not address:
            raise RuntimeError("RAY_ADDRESS unavailable")
        ray.init(address=address, namespace="vllm", ignore_reinit_error=True)
        cluster_resources = ray.cluster_resources()
        print(f"[ray-wait] cluster ready! resources: {cluster_resources}")
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
  # Ensure Ray environment is clean before vLLM starts
  echo "[vllm-pre] setting RAY_ADDRESS=$HEAD_IP:$RAY_PORT VLLM_HOST_IP=$HOST_IP"
  export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
  export VLLM_HOST_IP="$HOST_IP"
  unset RAY_GCS_SERVER_ADDRESS RAY_GCS_ADDRESS RAY_HEAD_NODE_IP \
        RAY_HEAD_SERVICE_PORT RAY_HEAD_MANAGER_PORT RAY_BOOTSTRAP_ADDRESS \
        RAY_CLUSTER_LAUNCHER_ADDRESS RAY_REDIS_ADDRESS RAY_HEAD_ADDRESS
  echo "[vllm-pre] final env: RAY_ADDRESS=$RAY_ADDRESS VLLM_HOST_IP=$VLLM_HOST_IP"
  cat > "$INFER_SCRIPT" <<'PY'
import os
import ray
from vllm import LLM, SamplingParams

def main() -> None:
    print(f"[vllm] starting inference with RAY_ADDRESS={os.environ.get('RAY_ADDRESS', 'unset')}")
    print(f"[vllm] IP configuration: VLLM_HOST_IP={os.environ.get('VLLM_HOST_IP', 'unset')} HOST_IP={os.environ.get('HOST_IP', 'unset')}")

    model_id = os.environ['MODEL_ID']
    model_local = f"/project/hf-cache/models/{model_id.replace('/', '-') }"
    out_dir = os.environ['OUTDIR']
    os.makedirs(out_dir, exist_ok=True)

    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        os.environ.setdefault('HUGGINGFACE_HUB_TOKEN', hf_token)

    hip_mask = os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('ROCR_VISIBLE_DEVICES')
    if not hip_mask:
        raise SystemExit('GPU visibility masks are missing inside worker environment')

    # Fix GPU environment variable conflicts for Ray workers
    # Ray manages ROCR_VISIBLE_DEVICES automatically, so ensure HIP uses the same devices
    rocr_devices = os.environ.get('ROCR_VISIBLE_DEVICES', hip_mask)
    if rocr_devices and rocr_devices != hip_mask:
        print(f"[vllm] adjusting HIP_VISIBLE_DEVICES from {hip_mask} to match ROCR_VISIBLE_DEVICES={rocr_devices}")
        os.environ['HIP_VISIBLE_DEVICES'] = rocr_devices
        os.environ['ROCM_VISIBLE_DEVICES'] = rocr_devices
        os.environ['CUDA_VISIBLE_DEVICES'] = rocr_devices

    print(f"[vllm] final GPU masks: HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")

    # Check Ray cluster state before vLLM init
    try:
        ray_address = os.environ.get('RAY_ADDRESS')
        if ray_address and not ray.is_initialized():
            print(f"[vllm] connecting to Ray at {ray_address}")
            ray.init(address=ray_address, ignore_reinit_error=True)
        resources = ray.cluster_resources()
        print(f"[vllm] Ray cluster resources: {resources}")
        nodes = ray.nodes()
        print(f"[vllm] Ray cluster has {len(nodes)} nodes")
        for i, node in enumerate(nodes):
            print(f"[vllm]   node {i}: alive={node['Alive']} resources={node.get('Resources', {})}")
    except Exception as e:
        print(f"[vllm] Ray cluster check failed: {e}")

    print(f"[vllm] creating LLM with TP={os.environ['TP']} PP={os.environ['PP']}")
    llm = LLM(
        model=model_local,
        tensor_parallel_size=int(os.environ['TP']),
        pipeline_parallel_size=int(os.environ['PP']),
        distributed_executor_backend='ray',
        disable_log_stats=True,
        trust_remote_code=True,
    )
    print("[vllm] LLM initialized successfully")

    prompt = (
        "You are running on the LUMI supercomputer. Write a single sentence "
        "announcing that multi-node vLLM inference works."
    )
    params = SamplingParams(temperature=0.7, max_tokens=80)
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text.strip()
    print("[inference] prompt:", prompt)
    print("[inference] completion:\n" + text)

    with open(os.path.join(out_dir, "response_rank0.txt"), "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


if __name__ == '__main__':
    main()
PY

  python "$INFER_SCRIPT"
  sentinel_put done inference.done
  ray stop --force || true
else
  sentinel_wait inference.done >/dev/null
  ray stop --force || true
fi
