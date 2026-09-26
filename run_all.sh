#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-all}
MARKETS=${MARKETS:-"NP PJM BE FR DE"}
MODELS=${MODELS:-"lear dnn epf_transformer transformer_imputed transformer_masked inr"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

train_all () {
  [ -f data/processed/DE_series.pkl ] || uv run python -m data.preprocess
  for m in $MARKETS; do
    for model in $MODELS; do
      seeds=$SEEDS
      variants="full ablated"
      [ "$model" = lear ] && seeds=0
      [ "$model" = inr ] && variants="full ablated static noexog"
      for s in $seeds; do
        for v in $variants; do
          uv run python train.py "$model" --market "$m" --seed "$s" --variant "$v"
        done
      done
    done
  done
}

eval_all () {
  uv run python evaluate.py
  uv run python tables.py
}

case "$STAGE" in
  train) train_all ;;
  eval)  eval_all ;;
  all)   train_all; eval_all ;;
  *)     echo "usage: $0 [all|train|eval]"; exit 1 ;;
esac
