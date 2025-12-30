#!/bin/bash
# Distributed training configuration

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


export OMP_NUM_THREADS=4
export NCCL_DEBUG=0
NPROC_PER_NODE=8

MASTER_ADDR=${1:-localhost}
NNODES=${2:-1}
NODE_RANK=${3:-0}

if [ -z "$4" ]; then
  MASTER_PORT=$(shuf -i 20001-29999 -n 1)
else
  MASTER_PORT=$4
fi


# === Launch ===
echo "Launching torchrun with:"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$NODE_RANK"
echo "NNODES=$NNODES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"

data_root=../
# DeepSpeed configuration
deepspeed=./train_scripts/zero2.json

# Model configuration
llm=$data_root/data/weights/Qwen2.5-VL-7B-Instruct   # Using HuggingFace model ID
tokenizers=$llm

lr=1e-5
batch_size=32
grad_accum_steps=1

datasets=egoman_pretrain

# output configuration
run_name="egoman7b_pretrain"
output_dir=$data_root/data/weights/$run_name

# Training arguments
args="
--deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --tokenizer_path "${tokenizers}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --lora_enable False \
    --bf16 \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --num_train_epochs 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.05 \
    --warmup_ratio 0.02 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard" \

torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
 tools/reason_pretrain.py ${args}
