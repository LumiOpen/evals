QUEUE="${QUEUE:-standard-g}"
PROJECT="${PROJECT:-${LUMI_PROJECT:?set PROJECT or LUMI_PROJECT}}"
LANGUAGE=${2:-en}

# IFEval, AlpacaEval, Multi-IF run in parallel
python3 main.py --time 10:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --model $1 ifeval
python3 main.py --time 4:00:00 --project $PROJECT --partition $QUEUE --model $1 alpaca_eval_$LANGUAGE

# Multi-IF only supports English
if [[ "$LANGUAGE" == "en" ]]; then
    python3 main.py --time 6:00:00 --project $PROJECT --partition $QUEUE --model $1 multi_if
fi

# MTBench inference
MTBENCH_OUTPUT=$(python3 main.py --time 4:00:00 --project $PROJECT --partition $QUEUE --model $1 mtbench_inference_$LANGUAGE)
echo "$MTBENCH_OUTPUT"

# Parse job ID and use as an afterok dependency for the judge task
if echo "$MTBENCH_OUTPUT" | grep -q "already exists and --force not specified, skipping"; then
    # Inference already completed, no dependency needed
    python3 main.py --time 2:00:00 --project $PROJECT --partition small --model $1 mtbench_judge_$LANGUAGE
elif echo "$MTBENCH_OUTPUT" | grep -q "Already detected job"; then
    MTBENCH_JOB_ID=$(echo "$MTBENCH_OUTPUT" | grep "Already detected job" | sed -n 's/.*job \([0-9]*\).*/\1/p')
    python3 main.py --time 2:00:00 --project $PROJECT --partition small --dependency afterok:$MTBENCH_JOB_ID --model $1 mtbench_judge_$LANGUAGE
else
    MTBENCH_JOB_ID=$(echo "$MTBENCH_OUTPUT" | grep '"job_id"' | sed -n 's/.*"job_id": *"\?\([0-9]*\)"\?.*/\1/p')
    python3 main.py --time 2:00:00 --project $PROJECT --partition small --dependency afterok:$MTBENCH_JOB_ID --model $1 mtbench_judge_$LANGUAGE
fi