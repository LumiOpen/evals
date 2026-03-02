#!/bin/bash

# Script to summarize RULER long context evaluation results across different sequence lengths

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIR>"
    echo ""
    echo "Example: $0 results/my-model"
    echo ""
    echo "This script summarizes RULER long context evaluation results."
    echo "It looks for files matching: ruler_<subtask>_<seqlen>.json or vllm_ruler_<subtask>_<seqlen>.json"
    exit 1
fi

DIR=$1

# RULER subtasks
NIAH_SINGLE=("niah_single_1" "niah_single_2" "niah_single_3")
NIAH_MULTI=("niah_multikey_1" "niah_multikey_2" "niah_multikey_3" "niah_multivalue" "niah_multiquery")
OTHER_TASKS=("ruler_vt" "ruler_cwe" "ruler_fwe" "ruler_qa_hotpot" "ruler_qa_squad")
ALL_SUBTASKS=("${NIAH_SINGLE[@]}" "${NIAH_MULTI[@]}" "${OTHER_TASKS[@]}")

# Sequence lengths (powers of 2 from 4K to 128K)
SEQ_LENGTHS=(4096 8192 16384 32768 65536 131072)
SEQ_LABELS=("4K" "8K" "16K" "32K" "64K" "128K")

get_result() {
    local file="$1"
    local subtask="$2"
    local seqlen="$3"

    # Treat missing OR empty files as no result.
    if [ -z "$file" ] || [ ! -s "$file" ]; then
        echo "na"
        return
    fi

    # Call jq directly (avoid cat|jq pipeline).
    local result
    result=$(jq -r "
        if .results[\"$subtask\"][\"${seqlen},none\"] then
            .results[\"$subtask\"][\"${seqlen},none\"] * 100
        elif .results[\"$subtask\"][\"acc,none\"] then
            .results[\"$subtask\"][\"acc,none\"] * 100
        elif .results[\"$subtask\"][\"exact_match,none\"] then
            .results[\"$subtask\"][\"exact_match,none\"] * 100
        elif .results[\"$subtask\"].acc then
            .results[\"$subtask\"].acc * 100
        elif .results[\"$subtask\"].exact_match then
            .results[\"$subtask\"].exact_match * 100
        else
            \"na\"
        end
    " "$file" 2>/dev/null)

    if [ "$result" = "null" ] || [ "$result" = "na" ] || [ -z "$result" ]; then
        echo "na"
    else
        printf "%.2f" "$result"
    fi
}

# Function to get result from a file
# Args: filename, subtask, seqlen
# get_result() {
#     local file="$1"
#     local subtask="$2"
#     local seqlen="$3"
    
#     if [ ! -f "$file" ]; then
#         echo "na"
#         return
#     fi
    
#     # RULER results use sequence length in the key name: "<seqlen>,none"
#     # Try multiple possible key formats
#     local result=$(cat "$file" | jq -r "
#         if .results[\"$subtask\"][\"${seqlen},none\"] then 
#             .results[\"$subtask\"][\"${seqlen},none\"] * 100
#         elif .results[\"$subtask\"][\"acc,none\"] then 
#             .results[\"$subtask\"][\"acc,none\"] * 100
#         elif .results[\"$subtask\"][\"exact_match,none\"] then 
#             .results[\"$subtask\"][\"exact_match,none\"] * 100
#         elif .results[\"$subtask\"].acc then 
#             .results[\"$subtask\"].acc * 100
#         elif .results[\"$subtask\"].exact_match then 
#             .results[\"$subtask\"].exact_match * 100
#         else 
#             \"na\" 
#         end
#     " 2>/dev/null)
    
#     if [ "$result" == "null" ] || [ "$result" == "na" ] || [ -z "$result" ]; then
#         echo "na"
#     else
#         printf "%.2f" "$result"
#     fi
# }

find_file() {
    local subtask="$1"
    local seqlen="$2"

    # Prefer the exact (legacy) filename when present.
    if [ -f "$DIR/ruler_${subtask}_${seqlen}.json" ]; then
        echo "$DIR/ruler_${subtask}_${seqlen}.json"
        return
    fi
    if [ -f "$DIR/vllm_ruler_${subtask}_${seqlen}.json" ]; then
        echo "$DIR/vllm_ruler_${subtask}_${seqlen}.json"
        return
    fi

    # Newer runners may append an ISO-like timestamp before the .json extension, e.g.
    # vllm_ruler_<subtask>_<seqlen>_2026-02-09T20-05-08.715047.json
    # Accept any extra suffix after the seqlen, and pick the newest match if multiple exist.
    local candidates=()

    # Match both underscore- and dash-separated suffix variants after <seqlen>.
    while IFS= read -r -d '' f; do candidates+=("$f"); done < <(find "$DIR" -maxdepth 1 -type f -name "ruler_${subtask}_${seqlen}_*.json" -print0 2>/dev/null)
    while IFS= read -r -d '' f; do candidates+=("$f"); done < <(find "$DIR" -maxdepth 1 -type f -name "ruler_${subtask}_${seqlen}-*.json" -print0 2>/dev/null)
    while IFS= read -r -d '' f; do candidates+=("$f"); done < <(find "$DIR" -maxdepth 1 -type f -name "vllm_ruler_${subtask}_${seqlen}_*.json" -print0 2>/dev/null)
    while IFS= read -r -d '' f; do candidates+=("$f"); done < <(find "$DIR" -maxdepth 1 -type f -name "vllm_ruler_${subtask}_${seqlen}-*.json" -print0 2>/dev/null)

    if [ ${#candidates[@]} -eq 0 ]; then
        echo ""
        return
    fi

    # Print most recently modified file.
    ls -1t "${candidates[@]}" 2>/dev/null | head -n 1
}

# Function to find the file (check both regular and vllm prefix)
# find_file() {
#     local subtask="$1"
#     local seqlen="$2"
    
#     if [ -f "$DIR/ruler_${subtask}_${seqlen}.json" ]; then
#         echo "$DIR/ruler_${subtask}_${seqlen}.json"
#     elif [ -f "$DIR/vllm_ruler_${subtask}_${seqlen}.json" ]; then
#         echo "$DIR/vllm_ruler_${subtask}_${seqlen}.json"
#     else
#         echo ""
#     fi
# }

# Print header
echo "=========================================="
echo "RULER Long Context Evaluation Summary"
echo "=========================================="
echo ""

# Check if any RULER files exist
found_any=false
for subtask in "${ALL_SUBTASKS[@]}"; do
    for seqlen in "${SEQ_LENGTHS[@]}"; do
        file=$(find_file "$subtask" "$seqlen")
        if [ -n "$file" ]; then
            found_any=true
            break 2
        fi
    done
done

if [ "$found_any" = false ]; then
    echo "No RULER results found in $DIR"
    echo ""
    echo "Expected files: ruler_<subtask>_<seqlen>.json or vllm_ruler_<subtask>_<seqlen>.json"
    echo "Example: ruler_niah_single_1_4096.json"
    exit 0
fi

# Print results by category
echo "NIAH Single Needle Tasks:"
echo "------------------------"
for subtask in "${NIAH_SINGLE[@]}"; do
    printf "%-20s" "$subtask:"
    for i in "${!SEQ_LENGTHS[@]}"; do
        seqlen="${SEQ_LENGTHS[$i]}"
        file=$(find_file "$subtask" "$seqlen")
        result=$(get_result "$file" "$subtask" "$seqlen")
        printf "%8s" "$result"
    done
    echo ""
done

echo ""
echo "NIAH Multi-Key/Value/Query Tasks:"
echo "---------------------------------"
for subtask in "${NIAH_MULTI[@]}"; do
    printf "%-20s" "$subtask:"
    for i in "${!SEQ_LENGTHS[@]}"; do
        seqlen="${SEQ_LENGTHS[$i]}"
        file=$(find_file "$subtask" "$seqlen")
        result=$(get_result "$file" "$subtask" "$seqlen")
        printf "%8s" "$result"
    done
    echo ""
done

echo ""
echo "Other RULER Tasks (VT, CWE, FWE, QA):"
echo "-------------------------------------"
for subtask in "${OTHER_TASKS[@]}"; do
    printf "%-20s" "$subtask:"
    for i in "${!SEQ_LENGTHS[@]}"; do
        seqlen="${SEQ_LENGTHS[$i]}"
        file=$(find_file "$subtask" "$seqlen")
        result=$(get_result "$file" "$subtask" "$seqlen")
        printf "%8s" "$result"
    done
    echo ""
done

echo ""
echo "=========================================="
echo "Average Scores by Sequence Length:"
echo "=========================================="
printf "%-20s" "Sequence Length:"
for i in "${!SEQ_LABELS[@]}"; do
    printf "%8s" "${SEQ_LABELS[$i]}"
done
echo ""
printf "%-20s" "Average Score:"

for i in "${!SEQ_LENGTHS[@]}"; do
    seqlen="${SEQ_LENGTHS[$i]}"
    
    # Collect all valid scores for this sequence length
    scores=()
    for subtask in "${ALL_SUBTASKS[@]}"; do
        file=$(find_file "$subtask" "$seqlen")
        result=$(get_result "$file" "$subtask" "$seqlen")
        if [ "$result" != "na" ]; then
            scores+=("$result")
        fi
    done
    
    # Calculate average if we have scores
    if [ ${#scores[@]} -gt 0 ]; then
        avg=$(printf '%s\n' "${scores[@]}" | awk '{sum+=$1; count++} END {if (count>0) printf "%.2f", sum/count; else print "na"}')
        printf "%8s" "$avg"
    else
        printf "%8s" "na"
    fi
done
echo ""

echo ""
echo "=========================================="
echo "Average Scores by Task Category:"
echo "=========================================="

# Calculate category averages across all sequence lengths
for category_name in "NIAH Single" "NIAH Multi" "Other Tasks" "Overall"; do
    printf "%-20s" "$category_name:"
    
    case "$category_name" in
        "NIAH Single")
            tasks=("${NIAH_SINGLE[@]}")
            ;;
        "NIAH Multi")
            tasks=("${NIAH_MULTI[@]}")
            ;;
        "Other Tasks")
            tasks=("${OTHER_TASKS[@]}")
            ;;
        "Overall")
            tasks=("${ALL_SUBTASKS[@]}")
            ;;
    esac
    
    for i in "${!SEQ_LENGTHS[@]}"; do
        seqlen="${SEQ_LENGTHS[$i]}"
        
        scores=()
        for subtask in "${tasks[@]}"; do
            file=$(find_file "$subtask" "$seqlen")
            result=$(get_result "$file" "$subtask" "$seqlen")
            if [ "$result" != "na" ]; then
                scores+=("$result")
            fi
        done
        
        if [ ${#scores[@]} -gt 0 ]; then
            avg=$(printf '%s\n' "${scores[@]}" | awk '{sum+=$1; count++} END {if (count>0) printf "%.2f", sum/count; else print "na"}')
            printf "%8s" "$avg"
        else
            printf "%8s" "na"
        fi
    done
    echo ""
done

echo ""

