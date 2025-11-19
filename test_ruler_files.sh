#!/bin/bash

# Quick diagnostic to see what RULER files exist and what the script is looking for

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIR>"
    exit 1
fi

DIR=$1

echo "Looking in directory: $DIR"
echo ""

# Show all vllm_ruler files
echo "=== Files matching vllm_ruler_*.json ==="
ls -1 "$DIR"/vllm_ruler_*.json 2>/dev/null | head -20
echo ""

# Show all ruler files (non-vllm)
echo "=== Files matching ruler_*.json (non-vllm) ==="
ls -1 "$DIR"/ruler_*.json 2>/dev/null | grep -v "^vllm_" | head -20
echo ""

# Test the find_file logic
echo "=== Testing find_file logic ==="
SUBTASKS=("niah_single_1" "niah_single_2" "ruler_vt")
SEQLEN=4096

for subtask in "${SUBTASKS[@]}"; do
    file1="$DIR/ruler_${subtask}_${SEQLEN}.json"
    file2="$DIR/vllm_ruler_${subtask}_${SEQLEN}.json"
    
    echo "Checking for subtask: $subtask at length $SEQLEN"
    echo "  Looking for: ruler_${subtask}_${SEQLEN}.json"
    if [ -f "$file1" ]; then
        echo "    ✓ Found (regular)"
    else
        echo "    ✗ Not found"
    fi
    
    echo "  Looking for: vllm_ruler_${subtask}_${SEQLEN}.json"
    if [ -f "$file2" ]; then
        echo "    ✓ Found (vllm)"
        
        # Try to extract a result using the correct format (with seqlen)
        echo "    Trying to extract result..."
        result=$(cat "$file2" | jq -r ".results[\"$subtask\"][\"${SEQLEN},none\"] // .results[\"$subtask\"][\"acc,none\"] // .results[\"$subtask\"][\"exact_match,none\"] // \"na\"" 2>/dev/null)
        echo "    Result (using ${SEQLEN},none): $result"
        
        # Show available keys
        echo "    Available result keys for $subtask:"
        cat "$file2" | jq -r ".results[\"$subtask\"] | keys[]" 2>/dev/null | head -10
    else
        echo "    ✗ Not found"
    fi
    echo ""
done

