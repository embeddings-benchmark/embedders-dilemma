#!/usr/bin/env bash
# Run a full LLM listwise-rerank sweep for one served model across all 4
# first-stages × (BRIGHT-7 + BEIR-5). Resumable via mteb ResultCache + usage
# sidecars (completed cells skip instantly). Safe to re-run after allocation cycling.
#
# Usage: run_llm_sweep.sh <GPU> <MODEL> <BASE_URL> <LOGFILE>
set -u
GPU="$1"; MODEL="$2"; BASE_URL="$3"; LOG="$4"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export BASE_URL MODEL TOKEN=EMPTY USE_STRICT_JSON=false MAX_CONCURRENCY=128
cd "$(dirname "$0")/../.."   # repo root

BRIGHT="BRIGHTBiology BRIGHTEarthScience BRIGHTEconomics BRIGHTPsychology BRIGHTRobotics BRIGHTStackoverflow BRIGHTSustainableLiving"
BEIR="FiQA2018 NFCorpus SciFact SCIDOCS TRECCOVID"

for FS in "bm25s|" \
          "BAAI/bge-large-en-v1.5|--batch-size 32" \
          "lightonai/GTE-ModernColBERT-v1|--batch-size 32" \
          "Qwen/Qwen3-Embedding-8B|--max-seq-length 8192 --attn-impl flash_attention_2 --batch-size 16"; do
  fs="${FS%%|*}"; extra="${FS##*|}"
  for BENCH in "$BRIGHT" "$BEIR"; do
    .venv/bin/python scripts/experiments/run_pipeline.py \
        --first-stage "$fs" --llm-rerank --tasks $BENCH \
        --top-k 100 --max-queries 100 $extra
  done
done
echo "DONE-SWEEP-${MODEL}"
