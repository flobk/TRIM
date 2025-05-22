#!/bin/bash

python main.py  --model Qwen/Qwen2.5-3B \
                --prune_method wanda_owl \
                --use_trim \
                --sparsity_ratio 0.7 \
                --task wikitext \
                --device cuda \
                --Lamda 0.12 \
                --Hyper_m 5.0 \
                --load_layer_ratio

python main.py  --model Qwen/Qwen2.5-3B \
                --prune_method wanda_alpha \
                --use_trim \
                --sparsity_ratio 0.7 \
                --task wikitext \
                --device cuda \
                --epsilon 0.1