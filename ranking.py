import torch
import torch.nn.functional as F
import numpy as np

def rank_features_observable_plus_one(
    predict_fn,
    query_x,
    query_mask,
    neighbors_x,
    candidate_indices,
    device='cpu'
):
    """
    Implements the "Observable + 1" ranking algorithm.
    
    Args:
        predict_fn: A function that takes (x, mask) and returns logits (N, C).
        query_x: Tensor of shape (1, D) containing the query instance (normalized).
        query_mask: Tensor of shape (1, D) containing the mask for the query (1=observed, 0=unobserved).
        neighbors_x: Tensor of shape (K, D) containing the retrieved neighbors (normalized).
        candidate_indices: List of feature indices (int) that are candidates for acquisition (i.e., currently unobserved).
        device: Device to run computations on.
        
    Returns:
        best_feature_idx: The index of the best feature to acquire.
    """
    if not candidate_indices:
        return None
        
    K = neighbors_x.shape[0]
    D = query_x.shape[1]
    
    scores = {}
    
    # Ensure inputs are on device
    query_x = query_x.to(device)
    query_mask = query_mask.to(device)
    neighbors_x = neighbors_x.to(device)
    
    for f_j in candidate_indices:
        # 1. Imputation Loop: Extract values v_ij from neighbors
        # v_ij are neighbors_x[:, f_j]
        imputed_values = neighbors_x[:, f_j] # Shape (K,)
        
        # 2. Synthetic Query Construction
        # We create K synthetic queries, one for each neighbor imputation
        synthetic_batch_x = query_x.repeat(K, 1) # (K, D)
        synthetic_batch_mask = query_mask.repeat(K, 1) # (K, D)
        
        # Update the candidate feature column
        synthetic_batch_x[:, f_j] = imputed_values
        synthetic_batch_mask[:, f_j] = 1.0
        
        # 3. Inference
        with torch.no_grad():
            logits = predict_fn(synthetic_batch_x, synthetic_batch_mask)
            probs = F.softmax(logits, dim=1)
            
    # 4. Score Aggregation (Per Neighbor)
        # Minimize entropy for each neighbor
        # Entropy = - sum(p * log(p))
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=1) # (K,)
        
        # Store entropy for this feature across all neighbors
        scores[f_j] = entropy
        
    # 5. Selection (Per Neighbor)
    # scores is Dict[feature_idx, tensor(K,)]
    # We want to find for each neighbor i, which feature j minimizes scores[j][i]
    
    votes = []
    for i in range(K):
        best_score = float('inf')
        best_feat = None
        
        for f_j, entropy_vec in scores.items():
            val = entropy_vec[i].item()
            if val < best_score:
                best_score = val
                best_feat = f_j
        
        votes.append((best_feat, 1.0 / (best_score + 1e-6))) # Score is inverse entropy
        
    return votes
