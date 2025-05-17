#!/bin/bash

# --dataset --output_dir --max_samples --temperature

CUDA_VISIBLE_DEVICES=0 python ../training_tool/src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../../../Models/Llama-2-7b-chat-hf \
    --adapter_name_or_path ../adapter/ \
    --dataset generation_test_250_0 \
    --dataset_dir ../training_tool/data/ \
    --template default \
    --finetuning_type lora \
    --output_dir ../combined_training_reproduce/generation_test_250_0_r4/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 2 \
    --max_samples 500 \
    --temperature 0.95 \
    --predict_with_generate \
    --loraplus_lr_ratio=16.0 \
