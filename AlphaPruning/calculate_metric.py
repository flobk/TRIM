"""
This script calculates the metric which can be later used to assign layerwise pruning ratios.
Code is adapted from AlphaPruning https://arxiv.org/html/2410.10912v1
"""
import os
import argparse
import numpy as np
from esd_utils import get_esd_metrics

def compute_and_save_metric(args):
    # Ensure the cache path for the metric exists
    os.makedirs(args.ww_metric_cache, exist_ok=True)
    # Extract model name from full path
    model_name = args.model.split('/')[-1]
    metric_filepath = os.path.join(args.ww_metric_cache, f"{model_name}_{args.ww_metric}.npy")
    if not os.path.exists(metric_filepath):
        print("Computing ESD metrics for each layer...")
        print("Modelname:", args.model)
        print("Metric:", args.ww_metric)
        print("Cache_dir:", args.cache_dir)
        metrics = get_esd_metrics(args.model, args.ww_metric, args.cache_dir)
        np.save(metric_filepath, metrics)
        print(f"Metrics saved to {metric_filepath}")
    else:
        print(f"Metrics file exists: {metric_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Save Layerwise Sparsity Ratios for AlphaPruning")
    parser.add_argument("--model", type=str, required=True, help="Path or identifier of the model")
    parser.add_argument("--cache_dir", type=str, default="llm_weights", help="Cache directory for model weights")
    parser.add_argument("--ww_metric", type=str, default="alpha_peak", help="Which metric to use (e.g. 'alpha_peak')")
    parser.add_argument("--ww_metric_cache", type=str, default="./metrics", help="Directory to cache the weightwatcher metric")
    args = parser.parse_args()

    compute_and_save_metric(args)

if __name__ == "__main__":
    main()