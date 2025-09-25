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
  singularity exec --rocm \
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
    "$IMG" bash /workspace/scripts/_inner.sh
