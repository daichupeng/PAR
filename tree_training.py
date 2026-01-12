# -*- coding: utf-8 -*-
import torch
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
import catboost as cb


class TreeMaskingPretrainer:
    '''
    Pretrain model with missing features using Tree models (CatBoost, XGBoost, RandomForest).
    Mirrors the functionality of MaskingPretrainer in crude_training.py but for tree models.
    '''
    def __init__(self, model_type='catboost', model_params=None, append_mask=True):
        self.model_type = model_type.lower()
        self.model_params = model_params if model_params else {}
        self.append_mask = append_mask
        self.model = None
        
        # Set default params if not provided
        if self.model_type == 'catboost' and not self.model_params:
            self.model_params = {'verbose': 0, 'allow_writing_files': False}
        
    def _prepare_data(self, loader):
        # Collect all data from loader
        X_list = []
        y_list = []
        
        for batch in loader:
            x, y = batch[0], batch[-1]
                
            X_list.append(x)
            y_list.append(y)
            
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        
        m = torch.ones(X.shape[0], X.shape[1])
            
        # Apply mask: x * m
        X_masked = X * m
        
        # Append mask if required
        if self.append_mask:
            X_final = torch.cat([X_masked, m], dim=1)
        else:
            X_final = X_masked
            
        return X_final.numpy(), y.numpy()

    def fit(self,
            train_loader,
            val_loader,
            verbose=True,
            num_classes=2
            ):
        '''
        Train the tree model.
        
        Args:
            n_epochs: For tree models, this is interpreted as the data augmentation factor.
                      Since trees don't train in epochs like NNs, we replicate the dataset 
                      n_epochs times with different masks to simulate the training process.
                      Default is 1 (no replication, just one pass of masking).
        '''
        
        if verbose:
            print(f"Preparing data for {self.model_type} training...")
            
        # Prepare training data with augmentation
        X_train_all = []
        y_train_all = []
        
        # Use n_epochs as augmentation factor
        # If n_epochs is very large (like 200 from crude_training), we should probably cap it 
        # or warn, as it will explode memory. 
        # For now, we'll assume the user passes a reasonable number or we cap it.
        # Let's cap at 10 for safety unless explicitly intended, but the user asked to mirror functions.
        # I'll use it as is but print a warning if > 10.
        
        X_tr, y_tr = self._prepare_data(train_loader)
        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
            
        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)
        
        # Prepare validation data (single pass)
        X_val, y_val = self._prepare_data(val_loader)
        
        if verbose:
            print(f"Training {self.model_type} model on {X_train.shape[0]} samples...")

        # Initialize and train model
        if self.model_type == 'catboost':
            if cb is None:
                raise ImportError("CatBoost is not installed. Please install it with `pip install catboost`.")
            
            # Auto-detect loss function based on num_classes if not specified
            params = self.model_params.copy()
            if 'loss_function' not in params:
                params['loss_function'] = 'MultiClass' if num_classes > 2 else 'Logloss'
            
            self.model = cb.CatBoostClassifier(**params)
                
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=verbose)
            
        elif self.model_type == 'xgboost':
            if xgb is None:
                raise ImportError("XGBoost is not installed. Please install it with `pip install xgboost`.")


            self.model = xgb.XGBClassifier(**self.model_params)

                
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
            
        elif self.model_type == 'randomforest':

            self.model = RandomForestClassifier(**self.model_params)

                
            self.model.fit(X_train, y_train)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Evaluate
        if verbose:
            print("Evaluating model...")
            
        preds = self.model.predict(X_val)
        
        acc = accuracy_score(y_val, preds)
        
        # Calculate F1 and AUROC
        if hasattr(self.model, "predict_proba"):
            preds_proba = self.model.predict_proba(X_val)
            
            if num_classes == 2:
                f1 = f1_score(y_val, preds)
                # For binary, roc_auc_score expects probability of positive class
                auc = roc_auc_score(y_val, preds_proba[:, 1])
            else:
                f1 = f1_score(y_val, preds, average='macro')
                auc = roc_auc_score(y_val, preds_proba, multi_class='ovr', average='macro')
        else:
            f1 = 0.0
            auc = 0.0
            if verbose:
                print("Warning: Model does not support predict_proba, skipping AUROC")

        if verbose:
            print(f"Validation Accuracy: {acc:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}")
        return acc, f1, auc


    def predict(self, x, m=None):
        '''
        Predict using the trained model.
        Args:
            x: Input features (torch.Tensor or numpy array)
            m: Mask (torch.Tensor or numpy array)
        '''
        if m is None:
            m = np.ones_like(x)

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
            
        x_masked = x * m
        if self.append_mask:
            x_final = np.concatenate([x_masked, m], axis=1)
        else:
            x_final = x_masked
            
        return self.model.predict(x_final)
    
    def predict_proba(self, x, m=None):
        '''
        Predict probabilities (for classification).
        '''
        if m is None:
            m = np.ones_like(x)
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"Model {self.model_type} does not support predict_proba")
            
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
            
        x_masked = x * m
        if self.append_mask:
            x_final = np.concatenate([x_masked, m], axis=1)
        else:
            x_final = x_masked
            
        return self.model.predict_proba(x_final)

    def apply_mask(self, x, m):
        '''
        Apply mask to input x.
        Args:
            x: Input features (torch.Tensor or numpy array)
            m: Mask (torch.Tensor or numpy array)
        Returns:
            x_final: Masked input (potentially with mask appended)
        '''
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
            
        x_masked = x * m
        if self.append_mask:
            x_final = np.concatenate([x_masked, m], axis=1)
        else:
            x_final = x_masked
        return x_final

def evaluate_tree_model(model_wrapper, dataloader, mask_percentage=0.0):
    '''
    Evaluates TreeMaskingPretrainer on a dataloader.
    
    Args:
        model_wrapper: The TreeMaskingPretrainer instance.
        dataloader: Yields (x, x_shap, y).
        mask_percentage: Percentage of features to mask (0.0 to 1.0).
                         If > 0, random mask is generated.
    '''
    all_preds = []
    all_probs = []
    all_targets = []
    
    for batch in dataloader:
        if len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch[0], batch[-1]
        
        # Convert to numpy
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
            
        N, D = x.shape
        
        # Generate mask
        if mask_percentage > 0:
            k = int(round(mask_percentage * D))
            m = np.ones_like(x)
            if k > 0:
                # Randomly mask k features per sample
                for i in range(N):
                    indices = np.random.choice(D, k, replace=False)
                    m[i, indices] = 0.0
        else:
            m = np.ones_like(x)
        
        # Predict probs (for AUROC)
        if hasattr(model_wrapper.model, "predict_proba"):
            probs = model_wrapper.predict_proba(x, m)
            all_probs.append(probs)
        else:
            # For models without predict_proba, store dummy
            all_probs.append(np.zeros((N, 2))) 
            
        # Predict labels (using predict directy ensures we get model labels, not just indices)
        preds = model_wrapper.predict(x, m)
        # Flatten if necessary (sometimes predict returns (N, 1))
        if preds.ndim > 1:
            preds = preds.ravel()
            
        all_preds.append(preds)
        
        if y.ndim > 1:
            y = y.ravel()
        all_targets.append(y)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    
    # Determine if binary or multiclass based on targets
    unique_targets = np.unique(all_targets)
    if len(unique_targets) == 2:
        # Binary prediction
        # Try binary F1 first (requires pos_label=1 usually, or labels 0, 1)
        try:
            f1 = f1_score(all_targets, all_preds)
        except ValueError:
            # Fallback if labels are weird
            f1 = f1_score(all_targets, all_preds, average='macro')
    else:
        f1 = f1_score(all_targets, all_preds, average='macro')
    
    # AUROC
    if hasattr(model_wrapper.model, "predict_proba"):
        if all_probs.shape[1] == 2:
            try:
                auc = roc_auc_score(all_targets, all_probs[:, 1])
            except ValueError:
                 auc = 0.0
        else:
            try:
                auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
            except ValueError:
                auc = 0.0
    else:
        auc = 0.0
        
    print(f'Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
        
    return acc, f1, auc
