import torch
import time
from typing import Dict, Tuple
import numpy as np

"""
This file contains utility functions for handling model pruning masks in PyTorch models.
It includes functionality to save, load and apply pruning masks to a hf model. 
The masks are stored in binary for optimal storage.
"""

def save_model_pruning_masks(model, save_path: str) -> None:
    """
    Save all pruning masks from a model to a single file using PyTorch's native functionality.
    
    Args:
        model: The pruned model
        save_path: Path to save the masks file
    """
    start_time = time.time()
    
    # Determine if it's OPT or another model type
    opt_flag = False
    if "OPT" in model.__class__.__name__:
        opt_flag = True
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    
    # Dictionary to store all masks and metadata
    all_masks = {}
    metadata = {
        "model_type": model.__class__.__name__,
        "is_opt": opt_flag,
        "num_layers": len(layers),
        "layer_info": {}
    }
    
    total_masks = 0
    total_params = 0
    total_pruned = 0
    
    # For each layer
    for i, layer in enumerate(layers):
        layer_info = {}
        
        # Find all linear layers
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                # Create mask where zeros in weight become True (pruned)
                mask = (weight == 0)
                
                if mask.any():  # Only save if there are pruned weights
                    # Pack the boolean mask into bits to save memory (8x reduction)
                    # and move to CPU immediately
                    packed_mask = torch.from_numpy(np.packbits(mask.cpu().numpy()))
                    
                    mask_key = f"layer{i}_{name}"
                    all_masks[mask_key] = packed_mask
                    
                    # Store original shape for unpacking later
                    original_shape = list(mask.shape)
                    
                    # Store metadata
                    num_params = mask.numel()
                    num_pruned = mask.sum().item()
                    sparsity = num_pruned / num_params
                    
                    layer_info[name] = {
                        "shape": original_shape,
                        "mask_key": mask_key,
                        "params": num_params,
                        "pruned": num_pruned,
                        "sparsity": float(sparsity),
                        "is_packed": True  # Flag to indicate this mask is bit-packed
                    }
                    
                    total_masks += 1
                    total_params += num_params
                    total_pruned += num_pruned
        
        metadata["layer_info"][str(i)] = layer_info
    print("Gathered all masks, saving to ", save_path)

    # Add summary statistics
    metadata["total_masks"] = total_masks
    metadata["total_params"] = total_params
    metadata["total_pruned"] = total_pruned
    metadata["overall_sparsity"] = float(total_pruned / total_params) if total_params > 0 else 0.0
    
    # Save all masks and metadata in a single file
    torch.save({
        "metadata": metadata,
        "masks": all_masks
    }, save_path)
    
    elapsed_time = time.time() - start_time
    print(f"Saved {total_masks} masks with {total_pruned}/{total_params} pruned weights ({metadata['overall_sparsity']*100:.2f}% sparsity)")
    print(f"Pruning masks saved to {save_path} in {elapsed_time:.2f} seconds")

def load_model_pruning_masks(load_path: str) -> Tuple[Dict, Dict]:
    """
    Load pruning masks from a single file using PyTorch's native functionality.
    
    Args:
        load_path: Path to the saved masks file
        
    Returns:
        Tuple of (metadata, masks_dict)
    """
    start_time = time.time()
    
    # Load the saved file
    saved_data = torch.load(load_path)
    metadata = saved_data["metadata"]
    packed_masks = saved_data["masks"]
    
    # Organize masks by layer and module
    organized_masks = {}
    for layer_idx, layer_info in metadata["layer_info"].items():
        organized_masks[layer_idx] = {}
        
        for name, module_info in layer_info.items():
            mask_key = module_info["mask_key"]
            packed_mask = packed_masks[mask_key]
            
            # Unpack the bit-packed mask to restore the boolean tensor
            original_shape = module_info["shape"]
            num_elements = 1
            for dim in original_shape:
                num_elements *= dim
            # Unpack the bits and reshape to the original tensor shape
            mask = torch.from_numpy(np.unpackbits(packed_mask.numpy())[:num_elements]).reshape(original_shape).bool()
                
            organized_masks[layer_idx][name] = mask
    
    elapsed_time = time.time() - start_time
    print(f"Loaded {metadata['total_masks']} masks with {metadata['overall_sparsity']*100:.2f}% overall sparsity")
    print(f"Masks loaded in {elapsed_time:.2f} seconds")
    
    return metadata, organized_masks

def apply_pruning_masks_to_model(model, masks_data) -> None:
    """
    Apply loaded pruning masks to a model using PyTorch's native functionality.
    
    Args:
        model: The model to apply masks to
        masks_data: Either a dictionary of masks organized by layer and module name,
                   or a tuple of (metadata, masks) as returned by load_model_pruning_masks
    """
    start_time = time.time()
    
    # Handle the case where masks_data is a tuple from load_model_pruning_masks
    if isinstance(masks_data, tuple) and len(masks_data) == 2:
        metadata, masks = masks_data
    else:
        masks = masks_data
    
    # Determine if it's OPT or another model type
    opt_flag = False
    if "OPT" in model.__class__.__name__:
        opt_flag = True
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    
    total_applied = 0
    
    # For each layer
    for layer_idx, layer_masks in masks.items():
        layer_idx = int(layer_idx)
        if layer_idx >= len(layers):
            print(f"Warning: Layer {layer_idx} not found in model")
            continue
        
        layer = layers[layer_idx]
        
        # For each module in the layer
        for module_name, mask in layer_masks.items():
            # Find the module
            module = None
            for name, mod in layer.named_modules():
                if name == module_name:
                    module = mod
                    break
            
            if module is None:
                print(f"Warning: Module {module_name} not found in layer {layer_idx}")
                continue
            
            # Apply the mask (set weights to zero where mask is True)
            module.weight.data[mask] = 0
            total_applied += 1
    
    elapsed_time = time.time() - start_time
    print(f"Applied {total_applied} pruning masks to model in {elapsed_time:.2f} seconds")