import torch 
import torch.nn as nn 
import os
import numpy as np
import pickle
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .mask_utils import save_model_pruning_masks
from .trim import calculate_row_wise_sparsities, prune_matrix

### https://arxiv.org/abs/2306.11695
def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

### https://arxiv.org/abs/2310.05175
def check_outlier_mean(mask,threshold):
    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()
    outlier_ratio=float(count)/total_params*100
    print("Outlier count: ", count, "Total params: ", total_params, "Outlier ratio: ", outlier_ratio)    
    return outlier_ratio

### https://arxiv.org/abs/2306.11695
def prepare_calibration_input(model, nsamples, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers        
    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    class Catcher_OPT(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
        
    if "OPT" in model.__class__.__name__:
        layers[0] = Catcher_OPT(layers[0])
    else:
        layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    inps = inps.cpu()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    if "OPT" in model.__class__.__name__:
        return inps, outs, attention_mask, None 
    else:
        position_ids = cache['position_ids']
        return inps, outs, attention_mask, position_ids 

### https://arxiv.org/abs/2306.11695
def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

#### https://arxiv.org/abs/2306.11695
def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None, trim=False):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples_calibration,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples_calibration, dataloader, device)
    print("prepared calibration input")

    opt_flag = False
    if "OPT" in model.__class__.__name__:
        opt_flag = True 
        layers=model.model.decoder.layers
        position_embeddings = None
    else: 
        # For llama3
        position_embeddings = model.model.rotary_emb(inps[0].unsqueeze(0), position_ids)
        layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    
    k=0
    all_sparsities = {}
    for i in range(len(layers)):
        print("-"*50, f"\n {i} / {len(layers)} \n", "-"*50)
        layer = layers[i]
        subset = find_layers(layer)
	
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = 'cuda'
        if device == 'cpu':
            print("Moving to GPU")
            layer = layers[i].to(dev)
        if position_embeddings:
            position_embeddings = (position_embeddings[0].to(dev), position_embeddings[1].to(dev))
        
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name], gather_inputs=trim) # gather inputs if necessary

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        print("attached hooks. Doing inference for pruning metric..")
        
        for j in range(args.nsamples_calibration):
            with torch.no_grad():
                inp_gpu = inps[j].unsqueeze(0).to(dev)
                if opt_flag: out = layer(inp_gpu, attention_mask=attention_mask)[0]
                else:        out = layer(inp_gpu, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                outs[j] = out.detach().cpu()  # Keep as tensor for next iteration
                del inp_gpu, out
                torch.cuda.empty_cache()
        for h in handles:
            h.remove()

        for name in subset:
            if trim: # TRIM
                ## unique layer ratio
                layer_sparsity_ratio= ratios[k]
                k += 1
                if layer_sparsity_ratio<=0:
                    layer_sparsity_ratio=0.01
                print("Sparsity ratio ", round(layer_sparsity_ratio,3), f"Layer {i} name {name}. Using TRIM." )
                ## Get wanda metric
                W = subset[name].weight.data.float().clone()
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                sorted_idx = torch.sort(W_metric, dim=-1, stable=True)[1]
                if args.input_recalc == True: # Recalc inputs if specified
                    ## For certain components, we need to collect input vectors again
                    if name in ["self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"] or name in ["self_attn.out_proj", "fc1", "fc2"]: # Adjust names if necessary
                        print("Do inference once more")
                        # First, clear existing input vectors
                        wrapped_layers[name].input_vectors = []
                        handles = []
                        handles.append(subset[name].register_forward_hook(add_batch(name)))
                        # Calc updated input vectors for mlp.up_proj at the same time
                        if name == "mlp.gate_proj":  # Adjust names if necessary
                            wrapped_layers["mlp.up_proj"].input_vectors = []
                            handles.append(subset["mlp.up_proj"].register_forward_hook(add_batch("mlp.up_proj")))
                        # Inference again
                        for j in range(args.nsamples_calibration):
                            with torch.no_grad():
                                inp_gpu = inps[j].unsqueeze(0).to(dev)
                                if opt_flag: out = layer(inp_gpu, attention_mask=attention_mask)[0]
                                else:        out = layer(inp_gpu, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                                outs[j] = out.detach().cpu()  # Keep as tensor for next iteration
                                del inp_gpu, out
                                torch.cuda.empty_cache()
                        for h in handles:
                            h.remove()
                ## Get input vectors
                inputs = torch.stack(wrapped_layers[name].input_vectors).to(dev).T.float()
                #### Calculate Individual Sparsites for each Row ####
                sparsity_list, parameter_dict = calculate_row_wise_sparsities(args, layer_sparsity_ratio, W, inputs, sorted_idx, dev)
                ## Dont allow sparsity > 95% for stability 
                sparsity_list[sparsity_list>0.95] = 0.95
                sparsity_list[sparsity_list<0] = 0
                sparsity_list = sparsity_list - sparsity_list.mean() + layer_sparsity_ratio
                all_sparsities[(i, name)] = (sparsity_list, parameter_dict) # Save sparsities and metrics
                ## Prune
                W_mask = ~prune_matrix(torch.ones_like(W_metric), sorted_idx, sparsity_list, dev).bool()
                subset[name].weight.data[W_mask] = 0  ## set weights to zero 
                del W, inputs
                torch.cuda.empty_cache()
                wrapped_layers[name].input_vectors = []
        
            else: # Normal wanda
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant 
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new 
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                        W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero 
    
        for j in range(args.nsamples_calibration):
            with torch.no_grad():
                inp_gpu = inps[j].unsqueeze(0).to(dev)
                if opt_flag: out = layer(inp_gpu, attention_mask=attention_mask)[0]
                else:        out = layer(inp_gpu, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                outs[j] = out.detach().cpu()  # Keep as tensor for next iteration
                del inp_gpu, out
                torch.cuda.empty_cache()
        inps, outs = outs, inps
	
        if device == 'cpu':
            print("Move back to CPU")
            layer = layer.to('cpu')

    print("-"*50)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    try:
        if args.sparsities_cache:
            model_name = args.model.split("/")[-1]
            job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
            filename = f"dimwise_sparsities_OWL_{model_name}_{args.sparsity_ratio}_{job_id}.pkl"
            with open(os.path.join(args.sparsities_cache, filename), "wb") as f:
                pickle.dump(
                    {"sparsities": all_sparsities,
                     "args": args}, f)
    except Exception as e:
        print(f"Saving sparsities has failed with error: {str(e)}")
    try:
        if args.save_masks: 
            model_name = args.model.split("/")[-1]
            job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
            filename = f"pruning_masks_{model_name}_{job_id}.pkl"
            save_model_pruning_masks(model, os.path.join(args.save_masks, filename))
    except Exception as e:
        print(f"Saving masks has failed with error: {str(e)}")

#### https://arxiv.org/abs/2310.05175
def prune_wanda_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, trim=False):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    import time
    t0 = time.time()
    if args.load_layer_ratio:
        model_name = args.model.split("/")[-1]
        all_layer_ratio = np.load(f"OWL/OWL_layer_ratios_{model_name}_{args.sparsity_ratio}_{args.Lamda}_{args.Hyper_m}.npy")
        print(f"OWL/OWL_layer_ratios_{model_name}_{args.sparsity_ratio}_{args.Lamda}_{args.Hyper_m}.npy")
    else:
        ##### calculate outlier ratio    
        all_layer_ratio=[]

        print("loading calibration data")
        dataloader, _ = get_loaders("c4",nsamples=args.nsamples_layerwise,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
        print("dataset loading complete")
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, args.nsamples_layerwise, dataloader, device)
        print("prepared calibration input")

        opt_flag = False
        if "OPT" in model.__class__.__name__:
            opt_flag = True 
            layers=model.model.decoder.layers
            position_embeddings = None
        else: 
            # For llama3
            position_embeddings = model.model.rotary_emb(inps[0].unsqueeze(0), position_ids)
            layers = model.model.layers

        for i in range(len(layers)):
            print("-"*50, f"\n {i} / {len(layers)}")
            layer = layers[i]
            subset = find_layers(layer)

            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
            else:
                dev = 'cuda'
            if device == 'cpu':
                print("Moving to GPU")
                layer = layers[i].to(dev)
            if position_embeddings:
                position_embeddings = (position_embeddings[0].to(dev), position_embeddings[1].to(dev))

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples_layerwise):
                with torch.no_grad():
                    inp_gpu = inps[j].unsqueeze(0).to(dev)
                    if opt_flag: out = layer(inp_gpu, attention_mask=attention_mask)[0]
                    else:        out = layer(inp_gpu, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                    outs[j] = out.detach().cpu()  # Keep as tensor for next iteration
                    del inp_gpu, out
                    torch.cuda.empty_cache()
            for h in handles:
                h.remove()
                
            layer_wmetric=[]
            for name in subset:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                layer_wmetric.append(W_metric)    
                    
            for j in range(args.nsamples_layerwise):
                with torch.no_grad():
                    inp_gpu = inps[j].unsqueeze(0).to(dev)
                    if opt_flag: out = layer(inp_gpu, attention_mask=attention_mask)[0]
                    else:        out = layer(inp_gpu, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                    outs[j] = out.detach().cpu()  # Keep as tensor for next iteration
                    del inp_gpu, out
                    torch.cuda.empty_cache()
            inps, outs = outs, inps
            if device == 'cpu':
                print("Move back to CPU")
                layer = layer.to('cpu')

            # Concat all layer component metrics into one layer metric array
            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
            
            for out_ratio in [args.Hyper_m]:
                out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
                print ("layer outlier ratio",out_ratio,out_ratio_layer)
            
            # append owl ratio for each layer
            all_layer_ratio.append(out_ratio_layer)
            
        # Right now all_layer_ratio is "LOD-vector" (https://arxiv.org/pdf/2310.05175)
        # Each score characterizes the outlier distribution of layer l
        # Take outlier score and calculate %-sparsity for each layer
        # lambda is hyperparameter that defines the range of the interval the sparsity falls in
        print ("*"*50,"\nbefore adjustment",all_layer_ratio)
        all_layer_ratio=np.array(all_layer_ratio)
        all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
        all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
        print("after adjustment",all_layer_ratio)
        print("mean",np.mean(all_layer_ratio), "max", np.max(all_layer_ratio), "min", np.min(all_layer_ratio))
        print("*"*50)

    if args.save_layer_ratio:
        model_name = args.model.split("/")[-1]
        np.save(f"OWL/OWL_layer_ratios_{model_name}_{args.sparsity_ratio}_{args.Lamda}_{args.Hyper_m}.npy", all_layer_ratio)
    all_layer_ratio = np.repeat(all_layer_ratio, len(find_layers(model.model)) // len(all_layer_ratio))
    print(f"OWL Time taken: {time.time() - t0:.2f} seconds")

    ### prune with layer specific sparsity
    print("Now pruning..")
    # wanda pruning
    prune_wanda(args, model, tokenizer, device, ratios=1-all_layer_ratio, prune_n=prune_n, prune_m=prune_m, trim=trim)

#### https://arxiv.org/abs/2410.10912
def prune_wanda_alpha(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, trim=False):
    ### Layer ratios using AlphaPruning
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon

    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # wanda pruning
    prune_wanda(args, model, tokenizer, device, ratios=all_layer_ratio, prune_n=prune_n, prune_m=prune_m, trim=trim)

#### https://arxiv.org/abs/2410.10912
def ww_sparsity(args, model, device=torch.device("cuda:0"), s1=0.8, s2=1.2):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    model_name = args.model.split('/')[-1]
    metric_filepath = os.path.join(args.ww_metric_cache, f"{model_name}_{args.ww_metric}.npy")
    metrics = np.load(metric_filepath)
    print("Loaded metrics")
    
    if args.mapping_type == 'block_wise':
        print("mapping blockwise")
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    
    print("metric values:", metrics)
            
    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    
    print("Layerwise pruning ratios: ",layerwise_pruning_ratios)
    return layerwise_pruning_ratios

def print_sparsity_information(model):
    """
    Reports sparsity for each layer, as well as globalwise
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("*"*50)
    
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    sparsity_count = 0 
    params_count = 0
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layer_sparsity_count = 0
        layer_param_count = 0
        for name in subset:
            W = subset[name].weight.data
            W_mask = (W==0)
            sparsities = W_mask.sum(dim=1) / W.shape[1]
            component_sp_ct = W_mask.sum().item()
            layer_sparsity_count += component_sp_ct
            layer_param_count += W.numel()
            if name[:9] == "self_attn": name = name[10:] # dont print selfattn prefix
            if name[:3] == "mlp": name = name[4:] # dont print mlp prefix
            print(f"{name:<10} {component_sp_ct:>10} {round(component_sp_ct/W.numel(),3):>10.3f} min:{sparsities.min().item():>7.3f} max:{sparsities.max().item():>7.3f} var{sparsities.var().item():>7.3f}")
        
        sparsity_count += layer_sparsity_count
        params_count += layer_param_count
        print()
        print(f"Layer {i:<4} {layer_sparsity_count:>10} {round(layer_sparsity_count/layer_param_count,3):>10.3f}")
        print("-"*50)

    model.config.use_cache = use_cache 
    print("Total N pruned: ",sparsity_count/1_000_000,"Million")
    print("Total Sparsity: ",round(float(sparsity_count)/params_count,4))
    print("*"*50)