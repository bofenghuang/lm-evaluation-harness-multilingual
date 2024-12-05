#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

set -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# tasks=leaderboard
# tasks=leaderboard_mmlu_pro
# tasks=leaderboard_musr

# tasks=leaderboard-fr
# tasks=leaderboard_musr_fr
# tasks=french_bench_arc_challenge
# tasks=french_bench_hellaswag
# tasks=french_bench
# tasks=belebele_fra_Latn

model_name_or_path=/projects/bhuang/models/llm/pretrained/meta-llama/Llama-3.2-1B-Instruct
# model_name_or_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
# model_name_or_path=bofenghuang/Llama-3-Vgn-8B-Instruct-v0.6
# model_name_or_path=OpenLLM-France/Claire-7B-FR-Instruct-0.1

tmp_model_id="$(echo "${model_name_or_path##*/}" | sed -e "s/[ |=-]/_/g" | tr '[:upper:]' '[:lower:]')"

gpus_per_model=1
model_replicas=4

    # max_model_len=4096
    # --log_samples \
    # --write_out \

lm_eval --model vllm \
    --model_args pretrained=${model_name_or_path},tensor_parallel_size=${gpus_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=${model_replicas},max_model_len=4096 \
    --tasks $tasks \
    --batch_size auto \
    --output_path results/${tasks} \

    # --num_fewshot 5 \

# python ./scripts/write_out.py \
#     --tasks $tasks \
#     --num_examples 5 \
#     --output_base_path results/samples/${tasks}

echo "END TIME: $(date)"
