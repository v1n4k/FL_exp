#!/usr/bin/env bash
# Central place to set all simulation configs; everything is forwarded to main.py.
set -euo pipefail
cd "$(dirname "$0")"

# ------------------------
# Editable configuration:
# NUM_CLIENTS      : number of simulated clients participating per round
# ROUNDS           : total federated rounds
# LOCAL_EPOCHS     : epochs of local fine-tuning per round on each client
# BATCH_SIZE       : local training batch size
# ALPHA            : Dirichlet concentration for non-IID label skew (smaller = more skewed)
# LR               : local optimizer learning rate
# MAX_LENGTH       : tokenizer max sequence length
# SEED             : global random seed for reproducibility
# INIT_NOISE_STD   : stddev of Gaussian noise added to LoRA params at client init (0 disables)
# OPTIMIZER        : sgd or adamw
# MOMENTUM         : momentum for SGD
# WEIGHT_DECAY     : weight decay for optimizer
# EARLY_STOP       : patience (rounds) for early stopping on personalized metric
# ORTHO_WEIGHT     : weight for orthogonality regularization on LoRA A (0 disables)
# USE_WANDB        : set to 1 to enable Weights & Biases logging
# WANDB_PROJECT    : wandb project name when USE_WANDB=1
# GPUS_PER_CLIENT  : number of GPUs to allocate per client (can be fractional)

NUM_CLIENTS=12
ROUNDS=30
LOCAL_EPOCHS=5
BATCH_SIZE=32
ALPHA=10
LR=3e-2
MAX_LENGTH=128
SEED=42
INIT_NOISE_STD=0.01
USE_WANDB=1  
WANDB_PROJECT="fedsa-fold"
GPUS_PER_CLIENT=2.0
OPTIMIZER="sgd"
MOMENTUM=0.9
WEIGHT_DECAY=0.0
EARLY_STOP=3
ORTHO_WEIGHT=0.1
# ------------------------

args=(
  --num-clients "$NUM_CLIENTS"
  --rounds "$ROUNDS"
  --local-epochs "$LOCAL_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --alpha "$ALPHA"
  --lr "$LR"
  --max-length "$MAX_LENGTH"
  --seed "$SEED"
  --init-noise-std "$INIT_NOISE_STD"
  --gpus-per-client "$GPUS_PER_CLIENT"
  --optimizer "$OPTIMIZER"
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --early-stop-patience "$EARLY_STOP"
  --orthogonal-reg-weight "$ORTHO_WEIGHT"
)

if [[ "$USE_WANDB" -ne 0 ]]; then
  args+=(--use-wandb --wandb-project "$WANDB_PROJECT")
fi

python main.py "${args[@]}" "$@"
