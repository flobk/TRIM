#!/bin/bash

# Use device = "cpu" to offload the model to memory
# cuda:0 device will then only be used for the layer in the current iteration
# This way only one GPU is necessary to prune 70B models with fp16.

# No tasks, device=cpu, save pruning masks
python main.py  --model Qwen/Qwen2.5-72B \
                --prune_method wanda_owl \
                --use_trim \
                --sparsity_ratio 0.7 \
                --task None \
                --device cpu \
                --Lamda 0.08 \
                --save_masks /mnt/lustre/work/... \
                --Hyper_m 7.0

python main.py  --model Qwen/Qwen2.5-72B \
                --prune_method wanda_alpha \
                --use_trim \
                --sparsity_ratio 0.7 \
                --task None \
                --device cpu \
                --save_masks /mnt/lustre/work/... \
                --epsilon 0.1