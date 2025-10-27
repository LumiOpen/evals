#!/bin/bash
# Run all HELMET tests for Poro2 and Llama 3.1 (base + instruct)
# Total: 24 jobs (4 models × 6 test types)

PROJECT=project_462000963
PARTITION=standard-g
GRES=gpu:mi250:4

echo "Queuing all HELMET tests for 4 models..."
echo "Total: 24 jobs"
echo ""

# ============================================================================
# PORO2 BASE - 6 tests
# ============================================================================
echo "=== Poro2 Base ==="

python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_recall_poro2_base

python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rag_short

python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rerank_short

python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_longqa_short

python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_summ_short

python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_icl_short

# ============================================================================
# PORO2 INSTRUCT - 6 tests
# ============================================================================
echo ""
echo "=== Poro2 Instruct ==="

python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_recall_poro2_instruct

python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rag_short

python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rerank_short

python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_longqa_short

python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_summ_short

python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_icl_short

# ============================================================================
# LLAMA 3.1 BASE - 6 tests
# ============================================================================
echo ""
echo "=== Llama 3.1 Base ==="

python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_recall_llama31_8b_base

python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rag_short

python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rerank_short

python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_longqa_short

python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_summ_short

python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_icl_short

# ============================================================================
# LLAMA 3.1 INSTRUCT - 6 tests
# ============================================================================
echo ""
echo "=== Llama 3.1 Instruct ==="

python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_recall_llama31_8b_instruct

python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rag_short

python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rerank_short

python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_longqa_short

python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_summ_short

python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_icl_short

echo ""
echo "=== All 24 jobs queued! ==="
echo "Monitor with: python watch.py"
