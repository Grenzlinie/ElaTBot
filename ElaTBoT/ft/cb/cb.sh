#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../training_tool/examples/accelerate/single_config.yaml \
    ../training_tool/src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ../../../Models/Llama-2-7b-chat-hf \
    --dataset combined_data \
    --dataset_dir ../training_tool/data/ \
    --template default \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../combined_training_reproduce \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --warmup_steps 20 \
    --save_steps 50000000 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --max_samples 50000000 \
    --val_size 0.05 \
    --upcast_layernor \
    --ddp_timeout 1800000 \
    --plot_loss \
    --fp16 \
    --loraplus_lr_ratio=16.0 \
