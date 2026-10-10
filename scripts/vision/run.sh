#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
set -a; [[ -f .env ]] && source .env; set +a

MODE="${MODE:-eval}"  # test or eval; can also set MODE=eval in the environment
VISION_GPUS="${VISION_GPUS:-1,2}"
VISION_BATCH_SIZE="${VISION_BATCH_SIZE:-64}"

case "$MODE" in
  test) DATA_DIR=multiagent/data/vision_test; LIMIT=2500 ;;
  eval) DATA_DIR=multiagent/data/vision_val; LIMIT=200 ;;
  *) echo "MODE must be test or eval (got: $MODE)" >&2; exit 1 ;;
esac

echo "MODE is $MODE"

VISION_GEMMA="${VISION_GEMMA:-${MULTIKV_MODELS_ROOT:?MULTIKV_MODELS_ROOT is not set; define it in .env}/google/gemma-3-4b-it}"
VISION_GRANITE="${VISION_GRANITE:-${MULTIKV_MODELS_ROOT:?MULTIKV_MODELS_ROOT is not set; define it in .env}/ibm-granite/granite-3.3-8b-instruct}"
VISION_RENDERER="${VISION_RENDERER:-multiagent/outputs_mine/swat/15E/stage2/exit-checkpoint-24000/renderer.pt}"

OUTPUT_DIR="${VISION_OUTPUT_DIR:-multiagent/outputs_mine/vision/vision_results}"
VISION_CAPTIONS="$OUTPUT_DIR/no_training/$MODE/gemma_captions.json"
TEMP_DIR="${VISION_TEMP_DIR:-multiagent/outputs/temp_vision_delete_me/$MODE}"

if [[ ! -f "$TEMP_DIR/plan.json" ]]; then
  ./env3.10/bin/python scripts/vision/eval.py prepare \
    --data-dir "$DATA_DIR" --limit "$LIMIT" \
    --work-dir "$TEMP_DIR" --shards 1
fi

./env3.10/bin/python scripts/vision/eval.py captions --gpus "$VISION_GPUS" \
  --data-dir "$DATA_DIR" --limit "$LIMIT" \
  --sender "$VISION_GEMMA" \
  --output "$VISION_CAPTIONS"

./env3.10/bin/python scripts/vision/eval.py baselines --gpus "$VISION_GPUS" \
  --data-dir "$DATA_DIR" --limit "$LIMIT" \
  --sender "$VISION_GEMMA" --receiver "$VISION_GRANITE" \
  --captions "$VISION_CAPTIONS" \
  --output-dir "$OUTPUT_DIR/no_training/$MODE" \
  --methods granite_only gemma_only t2t

./env3.10/bin/python scripts/vision/eval.py run --gpus "$VISION_GPUS" \
  --plan "$TEMP_DIR/plan.json" --input "$TEMP_DIR/input0.jsonl" \
  --sender "$VISION_GEMMA" --receiver "$VISION_GRANITE" \
  --renderer "$VISION_RENDERER" --captions "$VISION_CAPTIONS" \
  --output-dir "$OUTPUT_DIR/renderer/$MODE" \
  --methods captkv echo_v1 kvgain_v3 textscan_v6 scantrust_v2

./env3.10/bin/python scripts/vision/eval.py baselines --gpus "$VISION_GPUS" \
  --batch-size "$VISION_BATCH_SIZE" \
  --data-dir "$DATA_DIR" --limit "$LIMIT" \
  --sender "$VISION_GEMMA" --receiver "$VISION_GRANITE" \
  --captions "$VISION_CAPTIONS" \
  --ocr "$OUTPUT_DIR/renderer/$MODE/textscan_v6.json" \
  --output-dir "$OUTPUT_DIR/no_training/$MODE" \
  --methods gemma_s gemma_st gemma_cot

./env3.10/bin/python scripts/vision/eval.py baselines --gpus "$VISION_GPUS" \
  --data-dir "$DATA_DIR" --limit "$LIMIT" \
  --sender "$VISION_GEMMA" --receiver "$VISION_GRANITE" \
  --captions "$VISION_CAPTIONS" \
  --ocr "$OUTPUT_DIR/renderer/$MODE/textscan_v6.json" \
  --output-dir "$OUTPUT_DIR/no_training/$MODE" \
  --methods t2tplus

./env3.10/bin/python scripts/vision/eval.py run --gpus "$VISION_GPUS" \
  --plan "$TEMP_DIR/plan.json" --input "$TEMP_DIR/input0.jsonl" \
  --sender "$VISION_GEMMA" --receiver "$VISION_GRANITE" \
  --renderer "$VISION_RENDERER" --captions "$VISION_CAPTIONS" \
  --ocr "$OUTPUT_DIR/renderer/$MODE/textscan_v6.json" \
  --output-dir "$OUTPUT_DIR/renderer/$MODE" \
  --methods rand_captkv rand_echo_v1 rand_kvgain_v3 rand_textscan_v6 rand_scantrust_v2
