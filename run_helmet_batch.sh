#!/bin/bash
# Run HELMET tests in batches to avoid job submission limits
# Usage: bash run_helmet_batch.sh [batch_number]
# Batches: 1=Poro2Base, 2=Poro2Instruct, 3=Llama31Base, 4=Llama31Instruct

PROJECT=project_462000963
PARTITION=standard-g
GRES=gpu:mi250:4
BATCH=${1:-1}

case $BATCH in
  1)
    echo "=== Batch 1: Poro2 Base (6 jobs) ==="
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
    ;;

  2)
    echo "=== Batch 2: Poro2 Instruct (6 jobs) ==="
    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_recall_poro2_instruct

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_rag_short

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_rerank_short

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_longqa_short

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_summ_short

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_icl_short
    ;;

  3)
    echo "=== Batch 3: Llama 3.1 Base (6 jobs) ==="
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
    ;;

  4)
    echo "=== Batch 4: Llama 3.1 Instruct (6 jobs) ==="
    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_recall_llama31_8b_instruct

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_rag_short

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_rerank_short

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_longqa_short

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_summ_short

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      --apply_chat_template \
      helmet_icl_short
    ;;

  *)
    echo "Invalid batch number. Use 1-4:"
    echo "  1 = Poro2 Base"
    echo "  2 = Poro2 Instruct"
    echo "  3 = Llama 3.1 Base"
    echo "  4 = Llama 3.1 Instruct"
    exit 1
    ;;
esac

echo ""
echo "Batch $BATCH queued!"
echo "Monitor with: python watch.py"
