#!/bin/bash

# Test poro2-ropescaling128k checkpoint on all HELMET tasks
# Checkpoint: /scratch/project_462000963/users/jouni/CPT_from_finetuned/poro2-scripts-dev/converted-checkpoints/poro2-ropescaling128k/checkpoint_0001000

MODEL="/scratch/project_462000963/users/jouni/CPT_from_finetuned/poro2-scripts-dev/converted-checkpoints/poro2-ropescaling128k/checkpoint_0001000"
PROJECT="${PROJECT:-project_462000353}"
PARTITION="${PARTITION:-small-g}"
GRES="${GRES:-gpu:mi250:1}"

echo "Testing poro2-ropescaling128k on all HELMET tasks (8k, 16k, 32k)"
echo "Model: $MODEL"
echo ""

# Recall tasks
echo "=== Recall Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_recall_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_recall_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_recall_32k

# RAG tasks
echo "=== RAG Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rag_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rag_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rag_32k

# Rerank tasks
echo "=== Rerank Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rerank_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rerank_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_rerank_32k

# Long QA tasks
echo "=== Long QA Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_longqa_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_longqa_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_longqa_32k

# Summarization tasks
echo "=== Summarization Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_summ_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_summ_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_summ_32k

# In-context learning tasks
echo "=== ICL Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_icl_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_icl_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_icl_32k

# Citation tasks
echo "=== Citation Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_cite_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_cite_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_cite_32k

# ALCE No-Citation tasks
echo "=== ALCE No-Citation Tasks ==="
python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_alce_nocite_8k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_alce_nocite_16k

python main.py --model "$MODEL" --backend vllm \
  --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
  helmet_alce_nocite_32k

echo ""
echo "All 24 HELMET tests submitted (8 tasks × 3 context lengths)"
