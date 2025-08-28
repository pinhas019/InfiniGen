#!/usr/bin/env bash
set -euo pipefail

ROOT="/c/Users/morsh/Desktop/InfiniGen"
PY="$ROOT/.venv/Scripts/python.exe"

cd "$ROOT/accuracy"

mkdir -p "$ROOT/src"
cp -f "$ROOT/transformers/src/transformers/models/opt/modeling_opt_ours.py" "$ROOT/src/modeling_opt_ours.py"

export PYTHONPATH="$ROOT/transformers/src:$ROOT/src:$PYTHONPATH"

partial=0.2
seqlen=2048
alpha=4
budget=0.2

sizes=("125m") 
datasets=("wikitext2" "ptb")

for size in "${sizes[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "opt-$size $dataset 100% cache"
    "$PY" perplexity/opt.py --model "facebook/opt-${size}" \
      --eval_dataset "$dataset" \
      --seq_len "$seqlen" \
      --eval_samples 0 \
      --model_name "opt-${size}" \
      --infinigen \
      --partial_weight_ratio "$partial" \
      --partial_weight_path "$ROOT/setup/weights/opt-${size}_${partial}" \
      --alpha "$alpha" \
      --budget "$budget" \
      --capacity 1.0
  done
done

for size in "${sizes[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "opt-$size $dataset 80% cache evict arc"
    "$PY" perplexity/opt.py --model "facebook/opt-${size}" \
      --eval_dataset "$dataset" \
      --seq_len "$seqlen" \
      --eval_samples 0 \
      --model_name "opt-${size}" \
      --infinigen \
      --partial_weight_ratio "$partial" \
      --partial_weight_path "$ROOT/setup/weights/opt-${size}_${partial}" \
      --alpha "$alpha" \
      --budget "$budget" \
      --capacity 0.8 \
      --eviction_policy arc
  done
done
