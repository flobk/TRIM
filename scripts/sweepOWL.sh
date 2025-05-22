#!/bin/bash

# save layer ratio so that it can be loaded in the future with load_layer_ratio
for l in 0.02 0.05 0.08 0.12 0.15 0.20 
do 
python main.py  --model Qwen/Qwen2.5-3B \
                --prune_method wanda_owl \
                --sparsity_ratio 0.7 \
                --task wikitext \
                --device cuda \
                --seed 99 \
                --Lamda ${l} \
                --Hyper_m 5.0 \
                --save_layer_ratio
done