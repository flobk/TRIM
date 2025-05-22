import time
t_start = time.time()
from lib.eval import eval_ppl
from lib.eval import eval_zero_shot
import argparse
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_wanda_outlier, prune_wanda_alpha, print_sparsity_information

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, device="auto", cache_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
    torch_dtype=torch.float16, 
        cache_dir=cache_dir if cache_dir else None, 
        low_cpu_mem_usage=True, 
        device_map=device
    )
    # model.seqlen = model.config.max_position_embeddings
    model.seqlen = 2048
    return model

def main():
    parser = argparse.ArgumentParser()
    ########################## General Args ################################
    parser.add_argument('--model', type=str, help='facebook/opt-13b')
    parser.add_argument("--prune_method", type=str, help="Name in [wanda, wanda_owl, wanda_alpha]")
    parser.add_argument('--use_trim', action="store_true", help='Wether to use TRIM for calculating dimensionwise per-output sparsity.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level, float [0, 1]')
    parser.add_argument('--task', type=str, default='wikitext', help="Task in ['None', 'wikitext', 'both', 'all']")
    # optional
    parser.add_argument('--nsamples_calibration', type=int, default=256, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default=None, type=str, help="YOUR/MODEL/CACHE")
    parser.add_argument('--save_masks', type=str, default=None, help='Path to save the pruning masks.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--device', type=str, default="cuda", help='Device used. If "cpu", then model will be loaded to cpu, but cuda will still be used for inferencing the layer.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    
    #### TRIM Hyper parameters ####
    parser.add_argument('--qmetric', type=str, default="cosim_flat", help='Type of layer quality metric')
    parser.add_argument('--qmetric_dimwise', type=str, default="cosim", help='Type of dimensionwise quality metric')
    parser.add_argument('--forbid_neg_lr', action="store_true", help='Wether to forbid the algorithm to explore negative learning rates.')
    parser.add_argument('--input_recalc', action="store_true", help='Wether to recalc the input vectors for out and mlp projections.')
    parser.add_argument('--sparsities_cache', type=str, default=None, help='Only saves if arg is provided. Path to save the calculated dimensionwise-sparsities + metrics')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations during optimization')      
    
    #### OWL Hyper parameters ####
    parser.add_argument("--Lamda", default=0.05, type=float, help="Lamda for OWL layerwise sparsity range")
    parser.add_argument('--Hyper_m', type=float, default=7, help="Mean multiplier for OWL outlier detection")
    # optional
    parser.add_argument("--load_layer_ratio", action="store_true", help="Wether to load the layer ratio")
    parser.add_argument('--save_layer_ratio', action="store_true", help='Whether to save the layer ratio')
    parser.add_argument('--nsamples_layerwise', type=int, default=256, help='Number of calibration samples for layerwise sparsity allocation')
    parser.add_argument('--layer_ratio_cache', type=str, default="./OWL", help='Directory to cache the OWL layer ratio')
    
    #### AlphaPruning Hyper parameters ####
    parser.add_argument("--epsilon", type=float, default=0.3, help="Epsilon for mapping metric to pruning ratios") 
    # optional
    parser.add_argument("--ww_metric_cache", type=str, default="./AlphaPruning/metrics", help="Directory to cache the weightwatcher metric")
    parser.add_argument("--ww_metric", type=str, default="alpha_peak", help="Which metric to use (e.g. 'alpha_peak')")
    parser.add_argument("--mapping_type", default="block_wise", type=str, help="mapping type for pruning ratios allocation.")

    # Print all args
    args = parser.parse_args()
    print("*"*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("SLURM ID", os.environ.get('SLURM_JOB_ID', 'unknown'))
    print("*"*50)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # Loading model
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, cache_dir=args.cache_dir, device=args.device)
    print ("model is =================================================================================")
    model_name = args.model.split("/")[-1]
    print (model_name)
    print(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    for name, device in model.hf_device_map.items():
        print(f"{name}: device {device}")

    import time
    t_prune = time.time()
    ############################ baseline ############################
    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, prune_n=prune_n, prune_m=prune_m, device=args.device, trim=args.use_trim)
    elif args.prune_method == "wanda_owl":
        prune_wanda_outlier(args, model, tokenizer, prune_n=prune_n, prune_m=prune_m, device=args.device, trim=args.use_trim)
    elif args.prune_method == "wanda_alpha":
        prune_wanda_alpha(args, model, tokenizer, prune_n=prune_n, prune_m=prune_m, device=args.device, trim=args.use_trim)
    print("prune time: ", time.time() - t_prune)

    ### Print Sparsity ###
    print_sparsity_information(model)
    
    ### Testing ###
    t0 = time.time()
    # "wikitext" -> wikitext
    # "both"     -> wikitext + boolq
    # "all"      -> wikitext + boolq + rte + hellaswag + winogrande + arc_challenge + arc_easy + openbookqa
    if args.task in ["wikitext", "both", "all"]:
        ppl = eval_ppl(args, model, tokenizer, device)
        print(args.prune_method,f" Perplexity on wikitext {ppl}")
        print("wikitext time: ", time.time() - t0)
    if args.task in ["boolq", "both"]:
        accelerate=False
        task_list = ["boolq"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list=task_list, 
                                 num_fewshot=num_shot, use_accelerate=accelerate)
        print("zero_shot evaluation results")
        print(results)
        print("boolq time: ", time.time() - t0)
    if args.task in ["all"]:
        accelerate=False
        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list=task_list,
                                 num_fewshot=num_shot, use_accelerate=accelerate)
        print("zero_shot evaluation results") #(metrics only)
        print(results["results"])
        print("eval time: ", time.time() - t0)
    if args.save_model:
        print(args.save_model)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")

if __name__ == '__main__':
    main()
    print("Complete time:", time.time() - t_start)
