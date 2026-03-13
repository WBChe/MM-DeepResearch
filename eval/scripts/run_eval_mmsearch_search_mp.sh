#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DEV_ARR <<< "${CUDA_DEVICES}"
CHUNK_NUM="${#DEV_ARR[@]}"

SAVE_DIR="output_search/MMSearch"
NAME="test-8b-final"
INPUT_PATH="data/MMSearch_test.parquet"
OUTPUT_DIR="${SAVE_DIR}/${NAME}"

SEARCH_API_KEY=""
JINA_API_KEY=""

mkdir -p "${OUTPUT_DIR}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "chunk_num=${CHUNK_NUM}"
echo "output_dir=${OUTPUT_DIR}"

pids=()

for chunk_id in $(seq 0 $((CHUNK_NUM - 1))); do
  dev="${DEV_ARR[$chunk_id]}"
  shard_path="${OUTPUT_DIR}/shard_${chunk_id}_of_${CHUNK_NUM}.jsonl"

  echo "Launching chunk_id=${chunk_id} on GPU=${dev}"

  CUDA_VISIBLE_DEVICES="${dev}" \
  python3 eval.py \
    --input_path "${INPUT_PATH}" \
    --save_path "${shard_path}" \
    --idx_key data_id \
    --pool-size 1 \
    --rollout_times 1 \
    --enable_search \
    --engine serper \
    --search_api_key "${SEARCH_API_KEY}" \
    --jina_api_key "${JINA_API_KEY}" \
    --chunk_num "${CHUNK_NUM}" \
    --chunk_id "${chunk_id}" &

  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "[ERROR] At least one shard failed."
  exit 1
fi

echo "All shards finished."

OUT_PATH="${OUTPUT_DIR}/merged_${CHUNK_NUM}.jsonl"
: > "${OUT_PATH}"

for chunk_id in $(seq 0 $((CHUNK_NUM - 1))); do
  IN_PATH="${OUTPUT_DIR}/shard_${chunk_id}_of_${CHUNK_NUM}.jsonl"
  if [[ ! -f "${IN_PATH}" ]]; then
    echo "[WARN] missing ${IN_PATH}"
    continue
  fi
  cat "${IN_PATH}" >> "${OUT_PATH}"
done

echo "Merged into: ${OUT_PATH}"

python3 acc.py --save_path "${OUT_PATH}"