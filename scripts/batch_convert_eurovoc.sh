#!/bin/bash
# Batch convert all EuroVoc JSON files to HuggingFace Dataset format

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$SCRIPT_DIR/../data/eurovoc_filtered"
CONVERT_SCRIPT="$SCRIPT_DIR/convert_json_to_hf_dataset.py"

echo "=== Batch Converting EuroVoc JSON files to HF Dataset format ==="
echo "Data directory: $DATA_DIR"
echo

# Find all JSON files
json_files=("$DATA_DIR"/eurovoc_*.json)
total=${#json_files[@]}
count=0

for json_file in "${json_files[@]}"; do
    count=$((count + 1))

    # Extract basename without extension
    basename=$(basename "$json_file" .json)
    output_dir="$DATA_DIR/${basename}_dataset"

    # Skip if already converted
    if [ -d "$output_dir" ]; then
        echo "[$count/$total] SKIP: $basename (already exists)"
        continue
    fi

    echo "[$count/$total] Converting: $basename"
    python3 "$CONVERT_SCRIPT" "$json_file" "$output_dir"
    echo
done

echo "=== Conversion Complete ==="
echo "Verifying converted datasets..."
find "$DATA_DIR" -name "*_dataset" -type d | wc -l | xargs echo "Total datasets:"
