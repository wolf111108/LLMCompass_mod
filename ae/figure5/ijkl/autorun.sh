#!/usr/bin/env bash
set -u
set -o pipefail

# 假设这个脚本放在 ae/figure5/ijkl 目录下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

OUT_DIR="/home/zyzhao/LLMCompass/LLMCompass/ae/figure5/ijkl/output_qwen_prefill_init"
LOG_DIR="$OUT_DIR/logs"

rm -f "$OUT_DIR"/*.csv
rm -f "$OUT_DIR"/*.pdf
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT" || exit 1

export PATH=/home/zyzhao/.conda/envs/llmcompass_ae/bin:$PATH
export PYTHONUNBUFFERED=1

echo "=== Using Python: $(which python) ==="

CSV="$OUT_DIR/cim_dram_summary.csv"

echo "mode,array_height,array_width,Nbank,core_count,dram_read_value,dram_read_unit,dram_write_value,dram_write_unit,status,log_file" > "$CSV"

ARRAY_HEIGHT=64
NBANK=24
CORE_COUNT=16

WIDTH_LIST=(16 24 32 40 48 56 64 72 80 88 96)

echo "=== Step 1: CIM Prefill Init ==="

for ARRAY_WIDTH in "${WIDTH_LIST[@]}"; do
    echo
    echo "============================================================"
    echo "Running: array_height=${ARRAY_HEIGHT}, array_width=${ARRAY_WIDTH}, Nbank=${NBANK}, core_count=${CORE_COUNT}"
    echo "============================================================"

    LOG_FILE="$LOG_DIR/cim_init_opt_ah${ARRAY_HEIGHT}_aw${ARRAY_WIDTH}_nb${NBANK}_core${CORE_COUNT}.log"

    python -u -m ae.figure5.ijkl.test_transformer \
        --simcim \
        --init \
        --qwen \
        --array_height "$ARRAY_HEIGHT" \
        --array_width "$ARRAY_WIDTH" \
        --Nbank "$NBANK" \
        --core_count "$CORE_COUNT" \
        2>&1 | tee "$LOG_FILE"

    PY_STATUS=${PIPESTATUS[0]}

    DRAM_READ_VALUE=$(grep -E "^[[:space:]]*总 DRAM read:" "$LOG_FILE" | tail -n 1 | awk '{print $(NF-1)}')
    DRAM_READ_UNIT=$(grep -E "^[[:space:]]*总 DRAM read:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')

    DRAM_WRITE_VALUE=$(grep -E "^[[:space:]]*总 DRAM write:" "$LOG_FILE" | tail -n 1 | awk '{print $(NF-1)}')
    DRAM_WRITE_UNIT=$(grep -E "^[[:space:]]*总 DRAM write:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')

    if [ "$PY_STATUS" -ne 0 ]; then
        STATUS="FAIL"
    elif [ -z "$DRAM_READ_VALUE" ] || [ -z "$DRAM_WRITE_VALUE" ]; then
        STATUS="PARSE_FAIL"
    else
        STATUS="OK"
    fi

    echo "Result: read=${DRAM_READ_VALUE} ${DRAM_READ_UNIT}, write=${DRAM_WRITE_VALUE} ${DRAM_WRITE_UNIT}, status=${STATUS}"

    echo "init,${ARRAY_HEIGHT},${ARRAY_WIDTH},${NBANK},${CORE_COUNT},${DRAM_READ_VALUE},${DRAM_READ_UNIT},${DRAM_WRITE_VALUE},${DRAM_WRITE_UNIT},${STATUS},${LOG_FILE}" >> "$CSV"
done

cd "$OUT_DIR" || exit 1

echo
echo "============================================================"
echo "All done."
echo "Summary CSV: $CSV"
echo "Logs saved in: $LOG_DIR"
echo "============================================================"

column -s, -t "$CSV" 2>/dev/null || cat "$CSV"