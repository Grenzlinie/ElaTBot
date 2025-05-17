#!/bin/bash

# --dataset_dir --output_dir check --max_samples --temperature

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../LLaMA-Factory/examples/accelerate/single_config.yaml \
    ../LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../../Models/Llama-2-7b-chat-hf \
    --adapter_name_or_path ../adapter/ \
    --dataset ec_short_test_dataset \
    --dataset_dir ../LLaMA-Factory/data/ \
    --template default \
    --finetuning_type lora \
    --output_dir ../output_dir/test_result \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 2 \
    --max_samples 2000 \
    --temperature 0.01 \
    --predict_with_generate \
    --loraplus_lr_ratio=16.0 \