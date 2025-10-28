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
    echo "=== Batch 1: Poro2 Base (8 jobs) ==="
    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_recall_poro2_base

    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_rag_poro2

    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_rerank_poro2

    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_longqa_poro2

    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_summ_poro2

    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_icl_poro2

    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_cite_short

    python main.py --model LumiOpen/Llama-Poro-2-8B-base --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_alce_nocite_short
    ;;

  2)
    echo "=== Batch 2: Poro2 Instruct (8 jobs) ==="
    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_recall_poro2_instruct --apply_chat_template

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_rag_poro2 --apply_chat_template

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_rerank_poro2 --apply_chat_template

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_longqa_poro2 --apply_chat_template

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_summ_poro2 --apply_chat_template

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_icl_poro2 --apply_chat_template

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_cite_short --apply_chat_template

    python main.py --model LumiOpen/Llama-Poro-2-8B-instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_alce_nocite_short --apply_chat_template
    ;;

  3)
    echo "=== Batch 3: Llama 3.1 Base (8 jobs) ==="
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

    python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_cite_short

    python main.py --model meta-llama/Llama-3.1-8B --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_alce_nocite_short
    ;;

  4)
    echo "=== Batch 4: Llama 3.1 Instruct (8 jobs) ==="
    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_recall_llama31_8b_instruct --apply_chat_template

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_rag_short --apply_chat_template

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_rerank_short --apply_chat_template

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_longqa_short --apply_chat_template

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_summ_short --apply_chat_template

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_icl_short --apply_chat_template

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_cite_short --apply_chat_template

    python main.py --model meta-llama/Llama-3.1-8B-Instruct --backend vllm \
      --project $PROJECT --partition $PARTITION --gres $GRES --time 12:00:00 \
      helmet_alce_nocite_short --apply_chat_template
    ;;

  *)
    echo "Invalid batch number. Use 1-4:"
    echo "  1 = Poro2 Base (8 tests: recall, RAG, rerank, longQA, summ, ICL, cite, alce)"
    echo "  2 = Poro2 Instruct (8 tests with chat template)"
    echo "  3 = Llama 3.1 Base (8 tests)"
    echo "  4 = Llama 3.1 Instruct (8 tests with chat template)"
    exit 1
    ;;
esac

echo ""
echo "Batch $BATCH queued! (8 jobs)"
echo "Monitor with: python watch.py"
