import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
import faiss
import copy
from typing import List, Tuple, Dict, Any
from ranking import rank_features_observable_plus_one
# from tabular_dataset import get_dataset, data_split # No longer needed

class AFAVectorDatabase:
    def __init__(self, model: nn.Module, train_loader: DataLoader, features=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = None # Can be set in train_encoder
        
        # 1. Infer Features
        if features is not None:
            self.features = list(features)
        else:
            # Try to get features from the underlying dataset
            dataset = train_loader.dataset
            if hasattr(dataset, 'features'):
                self.features = list(dataset.features)
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'features'): # Handle Subset
                self.features = list(dataset.dataset.features)
            else:
                raise ValueError("Features list not provided and cannot be inferred from dataset.")

        # 2. Handle Data and Normalization Stats
        # We need access to the full training tensors for indexing and stats calculation
        # Try to access tensors directly if possible for efficiency
        dataset = train_loader.dataset
        if hasattr(dataset, 'tensors'):
            X_tensor = dataset.tensors[0]
            self.y_tensor = dataset.tensors[-1]
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'tensors'): # Handle Subset
             # Extract subset tensors
             indices = dataset.indices
             X_tensor = dataset.dataset.tensors[0][indices]
             self.y_tensor = dataset.dataset.tensors[-1][indices]
        else:
            # Fallback: Iterate loader to collect data (less efficient but generic)
            X_list, y_list = [], []
            for x, y in train_loader:
                X_list.append(x)
                y_list.append(y)
            X_tensor = torch.cat(X_list)
            self.y_tensor = torch.cat(y_list)

        # Calculate Mean/Std from data
        # self.mean = X_tensor.mean(dim=0)
        # self.std = X_tensor.std(dim=0)
        # self.std[self.std == 0] = 1
        
        # Store raw data for retrieval
        self.X_raw_tensor = X_tensor
        
        # Normalize internal tensors
        # self.X_norm_tensor = (X_tensor - self.mean) / self.std
        # User passes normalized inputs, so we use X_tensor directly as the "normalized" tensor
        # UPDATE: User now passes UNNORMALIZED inputs. We will normalize in build_index.
        self.X_norm_tensor = None
        
        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device.")
            
        self.model.to(self.device)
        self.index = None # Built later

    def _forward_model(self, x, mask):
        logits, emb = self.model(x, mask)
        # StudentEmbedder already normalizes embedding
        return logits, emb # No shap_logits for StudentEmbedder yet


    def build_index(self, mean: torch.Tensor, std: torch.Tensor, additional_loader: DataLoader = None):
        """
        Builds the Faiss index using the trained model and stored training data.
        
        Args:
            mean: Mean tensor for normalization.
            std: Std tensor for normalization.
            additional_loader: Optional DataLoader containing additional data to index.
                               Should yield (x, x_shap, y).
        """
        self.model.eval()
        print("Building Vector Index...")
        
        # Normalize stored raw data
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        
        # Ensure X_raw_tensor is on device or CPU as needed. Usually CPU for storage.
        # Normalize: (X - mean) / std
        # Do it on CPU to save GPU memory if dataset is large, or GPU if fast.
        # Let's do CPU for safety.
        mean_cpu = self.mean.cpu()
        std_cpu = self.std.cpu()
        
        self.X_norm_tensor = (self.X_raw_tensor.cpu() - mean_cpu) / std_cpu
        
        # If additional loader provided, append data to stored tensors
        if additional_loader:
            print("Processing additional data for indexing...")
            X_list, y_list = [], []
            
            # Iterate and collect
            # Assuming loader yields (x, shap, y)
            for batch in additional_loader:
                x = batch[0]
                y = batch[-1]
                
                X_list.append(x)
                y_list.append(y)
                
            if X_list:
                X_new = torch.cat(X_list)
                y_new = torch.cat(y_list)
                
                # Append to existing tensors (Raw)
                self.X_raw_tensor = torch.cat([self.X_raw_tensor.cpu(), X_new.cpu()])
                self.y_tensor = torch.cat([self.y_tensor.cpu(), y_new.cpu()])
                
                # Normalize new data and append to norm tensor
                X_new_norm = (X_new.cpu() - mean_cpu) / std_cpu
                self.X_norm_tensor = torch.cat([self.X_norm_tensor, X_new_norm])
                
                print(f"Added {len(X_new)} additional vectors to index.")

        with torch.no_grad():
            x_tensor = self.X_norm_tensor.to(self.device)
            mask_tensor = torch.ones_like(x_tensor).to(self.device)
            
            outputs = self._forward_model(x_tensor, mask_tensor)
            _, embeddings = outputs
            embeddings = embeddings.cpu().numpy()
            
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")

    def retrieve_context(self, partial_instance: Dict[str, float], k: int = 3,) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
            
        # 1. Preprocess Input
        # print(partial_instance)
        x_vec = np.zeros((1, len(self.features)))
        mask_vec = np.zeros((1, len(self.features)))
        
        observed_features = set(partial_instance.keys())
        # print(observed_features)
        
        for feat, val in partial_instance.items():
            if feat in self.features:
                idx = self.features.index(feat)
                # Normalize
                # Note: self.mean and self.std are tensors
                # x_vec[0, idx] = (val - self.mean[idx].item()) / self.std[idx].item()
                # Assume input is already normalized
                x_vec[0, idx] = val
                mask_vec[0, idx] = 1.0
        
        # print(x_vec)
        # 2. Encode Partial View
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_vec).to(self.device)
            m_tensor = torch.FloatTensor(mask_vec).to(self.device)
            _,query_emb = self._forward_model(x_tensor, m_tensor)
            
        # 3. Search Index
        # print('Search Index...')
        # Faiss requires float32 and C-contiguous arrays
        query_emb = np.ascontiguousarray(query_emb.cpu().numpy(), dtype=np.float32)
        D, I = self.index.search(query_emb, k)
        
        # 4. Format Output
        retrieved_indices = I[0]
        retrieved_distances = D[0]
        results = []
        
        # Determine suggested feature using requested algorithm
        obs_plus_one_votes = []
        nrg_votes = []
        
        # Define predict_fn wrapper (common for both)
        def predict_fn(x, mask):
            # Check model type for correct call
            if hasattr(self.model, 'retrieval_head'): # StudentEmbedder
                 logits, _ = self.model(x, mask)
                 return logits

        # Identify candidate indices (unobserved)
        candidate_indices = [i for i, m in enumerate(mask_vec[0]) if m == 0]
        
        # Get neighbors vectors
        valid_indices = [idx for idx in retrieved_indices if idx != -1]
        
        if valid_indices and candidate_indices:
            neighbors_x = self.X_norm_tensor[valid_indices].to(self.device)

            obs_plus_one_indices = rank_features_observable_plus_one(
                predict_fn=predict_fn,
                query_x=x_tensor,
                query_mask=m_tensor,
                neighbors_x=neighbors_x,
                candidate_indices=candidate_indices,
                device=self.device
            )
            
            # Map indices to feature names and scores
            obs_plus_one_votes = []
            for idx, score in obs_plus_one_indices:
                feat_name = self.features[idx] if idx is not None else None
                obs_plus_one_votes.append((feat_name, score))
            

        for rank, idx in enumerate(retrieved_indices):
            if idx == -1: continue # Safety
            
            row_y = self.y_tensor[idx].item()
            
            # 4. Construct Result
            # We need to return the FULL features for the retrieved case
            # Return ORIGINAL (unnormalized) features as requested.
            row_vals_raw = self.X_raw_tensor[idx]
            
            row_features = {}
            for i, val in enumerate(row_vals_raw):
                feat_name = self.features[i]
                row_features[feat_name] = val.item()
            
            # Determine suggested feature
            suggested_feature = None
            feature_score = 0.0
            
            if obs_plus_one_votes:
                try:
                    vote_idx = valid_indices.index(idx)
                    suggested_feature, feature_score = obs_plus_one_votes[vote_idx]
                except ValueError:
                    suggested_feature = None
        
            
            results.append({
                "case_id": int(idx),
                "case_label": row_y,
                "suggested_feature": suggested_feature,
                "feature_score": feature_score,
                "distance": 1 - D[0][rank] if self.index.metric_type == faiss.METRIC_INNER_PRODUCT else D[0][rank], # Approximate distance from similarity if IP
                "similarity": retrieved_distances[rank],
                "full_features": row_features
            })
            
        return results

    def predict(self, observed_dict_norm: Dict[str, float]) -> torch.Tensor:
        """
        Returns the model output for the given observed features.
        
        Args:
            observed_dict_norm: Dictionary of observed feature values (normalized).
            
        Returns:
            Model output (logits for MLP, or embedding for Encoder).
        """
        # Preprocess
        x_vec = np.zeros((1, len(self.features)))
        mask_vec = np.zeros((1, len(self.features)))
        
        for feat, val in observed_dict_norm.items():
            if feat in self.features:
                idx = self.features.index(feat)
                x_vec[0, idx] = val
                mask_vec[0, idx] = 1.0
                
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_vec).to(self.device)
            m_tensor = torch.FloatTensor(mask_vec).to(self.device)
            
            if hasattr(self.model, 'retrieval_head'): # StudentEmbedder
                 logits, _ = self.model(x_tensor, m_tensor)
                 return logits, None # No shap_logits
