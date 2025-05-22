import torch

def calculate_row_wise_sparsities(args, target_sparsity, W, inputs, sorted_idx, device):
    """
    Iterative metric driven pruning.
    Finds an optimal sparsity vector that contains non-uniform sparsity ratios for individual output dimensions of W.
    """
    # Start with uniform sparsity
    best_sparsities = torch.ones(W.shape[0], device=device)*target_sparsity
    unpruned_output = W @ inputs
    pruned_output = prune_matrix(W, sorted_idx, best_sparsities, device) @ inputs
    # Track metrics
    baseline_metric = calc_qmetric(unpruned_output, pruned_output, args.qmetric)
    best_metric, best_learning_rate, best_iter, start_var = baseline_metric, None, 0, cosine_similarity_dimwise(unpruned_output, pruned_output).var().item()
    best_this_method = float("-inf")
    # explore pos lr
    for learning_rate in [0.01, 0.02, 0.04, 0.08, 0.12, 0.16]:
        sparsities, metric_score, iter = get_sparsities_iterative(args, target_sparsity, W, inputs, sorted_idx, device, learning_rate=learning_rate)
        if metric_score > best_metric:
            best_metric = metric_score
            best_sparsities = sparsities
            best_learning_rate = learning_rate
            best_iter = iter
        if metric_score > best_this_method: # break if no better lr is found
            best_this_method = metric_score
        else: break
    # explore neg lr
    if best_learning_rate == None and args.forbid_neg_lr != True:
        best_this_method = float("-inf")
        for learning_rate in [-0.01, -0.02, -0.04]:
            sparsities, metric_score, iter = get_sparsities_iterative(args, target_sparsity, W, inputs, sorted_idx, device, learning_rate=learning_rate)
            if metric_score > best_metric:
                best_metric = metric_score
                best_sparsities = sparsities
                best_learning_rate = learning_rate
                best_iter = iter
            if metric_score > best_this_method: # break if no better lr is found
                best_this_method = metric_score
            else: break
    print(f"Best metric score: {best_metric}, Baseline metric: {baseline_metric}, Learning rate: {best_learning_rate}")
    print("-"*30)
    end_var = cosine_similarity_dimwise(unpruned_output, prune_matrix(W, sorted_idx, best_sparsities, device) @ inputs).var().item()
    return best_sparsities, {"baseline_metric": baseline_metric, "best_metric": best_metric, "learning_rate": best_learning_rate, "best_iter": best_iter, "starting_var": start_var, "ending_var": end_var}

def get_sparsities_iterative(args, target_sparsity, W, inputs, sorted_idx, device, learning_rate=0.01):
    def update_x(x):
        """
        Updates x based on the qmetric_dimwise between unpruned and pruned outputs.
        """
        qm_dimwise = calc_qmetric_dimwise(unpruned_outputs, pruned_outputs, qmetric_dimwise)   
        x_d = (qm_dimwise - qm_dimwise.min()) / (qm_dimwise.max() - qm_dimwise.min() + 1e-6)
        x_d *= learning_rate * 2
        x_d = x_d - x_d.mean()
        if torch.isnan(x_d).any():
            print(f"Warning: Detected {torch.isnan(x_d).sum().item()} NaN values in COSIM x_d")
        x += x_d
        x[x>0.95] = 0.95
        x[x<0] = 0
        x = x - x.mean() + target_sparsity
        return x

    qmetric, qmetric_dimwise, iterations = args.qmetric, args.qmetric_dimwise, args.iterations
    # Calculate once in the beginning
    unpruned_outputs = W @ inputs
    # Initialize with normal (uniform) prune
    x = torch.ones(W.shape[0], device=device) * target_sparsity
    W_pruned = prune_matrix(W, sorted_idx, x, device)
    pruned_outputs = W_pruned @ inputs
    # Track metrics
    best_metric = calc_qmetric(unpruned_outputs, pruned_outputs, qmetric)
    best_x = x.clone()
    best_iter = 0
    for i in range(iterations):
        x = update_x(x)
        # Calculate new scores
        W_pruned = prune_matrix(W, sorted_idx, x, device)
        pruned_outputs = W_pruned @ inputs
        metric_score = calc_qmetric(unpruned_outputs, pruned_outputs, qmetric)
        # Track best x based on metric
        if (metric_score > best_metric) or qmetric == "pass":
            best_metric = metric_score
            best_x = x.clone()
            best_iter = i
    return best_x, best_metric, best_iter

def prune_matrix(W, sorted_idx, sparsities, device):
    """Efficiently prunes a matrix with specific sparsities for each row"""
    batch_size, width = W.shape
    mask = torch.ones_like(W, dtype=torch.bool)
    num_prune = (width * sparsities).long()  # [batch_size]
    # Create indices tensor for all rows at once
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, width)
    prune_indices = torch.arange(width, device=device).unsqueeze(0).expand(batch_size, -1)
    prune_mask = prune_indices < num_prune.unsqueeze(1)
    mask[row_indices, sorted_idx] = ~prune_mask
    return W * mask

def calc_qmetric(a,b, metric_type):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    if metric_type == "cosim_flat":
        return cosine_similarity(a,b).item()
    elif metric_type == "cosim_dim":
        return cosine_similarity_dimwise(a,b).mean().item()
    elif metric_type == "cosim_sample":
        return cosine_similarity_samplewise(a,b).mean().item()
    elif metric_type == "mse":
        return -1 * mean_squared_error(a,b).cpu().item() # bigger is better
    elif metric_type == "mse_dim":
        return -1 * mean_squared_error_dimwise(a,b).mean().item() # bigger is better
    elif metric_type == "mae":
        return -1 * mean_absolute_error(a,b) # bigger is better
    elif metric_type == "snr":
        return signal_to_noise_ratio(a,b) # bigger is better
    elif metric_type == "psnr_flat":
        return peak_signal_to_noise_ratio(a,b) # bigger is better
    elif metric_type == "psnr_dim":
        psnr_values = peak_signal_to_noise_ratio_dimwise(a, b)
        return psnr_values.mean().item() # bigger is better
    elif metric_type == "psnr_sample":
        psnr_values = peak_signal_to_noise_ratio_samplewise(a, b)
        return psnr_values.mean().item() # bigger is better
    elif metric_type == "pass":
        return 1

def calc_qmetric_dimwise(a,b, metric_type):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    if metric_type == "cosim":
        return cosine_similarity_dimwise(a,b)
    elif metric_type == "mse":
        return -1 * mean_squared_error_dimwise(a,b)
    elif metric_type == "psnr":
        psnr_values = peak_signal_to_noise_ratio_dimwise(a, b)
        return psnr_values
    elif metric_type == "pass":
        return 1


def mean_squared_error(a, b):
    """Compute mean squared error between tensors a and b"""
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    mse = torch.mean((a - b) ** 2)
    return mse

def mean_squared_error_dimwise(a, b):
    """Compute mean squared error along dimension 1"""
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    mse = torch.mean((a - b) ** 2, dim=1)
    return mse

def cosine_similarity(a, b):
    """Compute cosine similarity for flattened vectors with improved numerical stability"""
    # Clip the final result to [-1, 1] range
    a_norm = a / (torch.norm(a) + 1e-8)  # Add small epsilon to avoid division by zero
    b_norm = b / (torch.norm(b) + 1e-8)
    return torch.clamp(torch.sum(a_norm * b_norm), -1.0, 1.0)

def cosine_similarity_dimwise(a, b):
    """Compute cosine similarity along dimension 1 with improved numerical stability"""
    # Clip the final result to [-1, 1] range
    a_norm = a / (torch.norm(a, dim=1, keepdim=True) + 1e-8)  # Add small epsilon to avoid division by zero
    b_norm = b / (torch.norm(b, dim=1, keepdim=True) + 1e-8)
    return torch.clamp(torch.sum(a_norm * b_norm, dim=1), -1.0, 1.0)

def cosine_similarity_samplewise(a, b):
    """Compute cosine similarity along dimension 1 with improved numerical stability"""
    # Clip the final result to [-1, 1] range
    a_norm = a / (torch.norm(a, dim=0, keepdim=True) + 1e-8)  # Add small epsilon to avoid division by zero
    b_norm = b / (torch.norm(b, dim=0, keepdim=True) + 1e-8)
    return torch.clamp(torch.sum(a_norm * b_norm, dim=0), -1.0, 1.0)

def mean_absolute_error(a, b):
    """Compute mean absolute error between tensors a and b"""
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    mae = torch.mean(torch.abs(a - b))
    return mae.item() # Return as scalar

def signal_to_noise_ratio(a, b):
    """Compute Signal-to-Noise Ratio (SNR)"""
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    signal_power = torch.mean(a ** 2)
    noise_power = torch.mean((a - b) ** 2)
    if noise_power < 1e-10: # Avoid division by zero or log(0)
        return float('inf') # Or a very large number
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item() # Return as scalar

def peak_signal_to_noise_ratio(a, b, data_range=None):
    """Compute Peak Signal-to-Noise Ratio (PSNR)"""
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    mse = torch.mean((a - b) ** 2)
    # Removed near-zero check here to align with dim/samplewise versions' direct calculation
    # Original check: if mse < 1e-10: return float('inf')
    if data_range is None:
        # Use max absolute value from the original signal if range not specified
        peak_signal_sq = torch.max(a**2)
    else:
        peak_signal_sq = data_range ** 2
    # Add epsilon directly in the denominator to prevent division by zero / log(0)
    psnr = 10 * torch.log10(peak_signal_sq / (mse + 1e-10))
    return psnr.item() # Return as scalar

def peak_signal_to_noise_ratio_dimwise(a, b, data_range=None):
    """Compute Peak Signal-to-Noise Ratio (PSNR) along dimension 1"""
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    if a.dim() < 2:
         raise ValueError("Input tensors must have at least 2 dimensions for dimwise calculation")

    mse = torch.mean((a - b) ** 2, dim=1) # Shape: [batch_size]

    if data_range is None:
        peak_signal_sq = torch.max(a**2, dim=1)[0] # Shape: [batch_size]
    else:
        # Let broadcasting handle scalar data_range
        peak_signal_sq = torch.tensor(data_range ** 2, device=a.device, dtype=a.dtype)

    # Calculate PSNR, add epsilon to prevent division by zero / log(0)
    psnr = 10 * torch.log10(peak_signal_sq / (mse + 1e-10))

    return psnr # Return tensor of PSNR values for each row

def peak_signal_to_noise_ratio_samplewise(a, b, data_range=None):
    """Compute Peak Signal-to-Noise Ratio (PSNR) along dimension 0"""
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    if a.dim() < 1: # Dim 0 always exists unless scalar
         raise ValueError("Input tensors must have at least 1 dimension")

    mse = torch.mean((a - b) ** 2, dim=0) # Shape: [width]

    if data_range is None:
        peak_signal_sq = torch.max(a**2, dim=0)[0] # Shape: [width]
    else:
        peak_signal_sq = torch.tensor(data_range ** 2, device=a.device, dtype=a.dtype)

    # Calculate PSNR, add epsilon to prevent division by zero / log(0)
    psnr = 10 * torch.log10(peak_signal_sq / (mse + 1e-10))

    return psnr # Return tensor of PSNR values for each column/sample