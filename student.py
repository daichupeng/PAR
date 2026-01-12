# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
import copy
from torchmetrics import Accuracy, AUROC, F1Score
from sklearn.metrics import f1_score, roc_auc_score


def generate_uniform_mask(batch_size, num_features, device='cpu'):
    '''Generate binary masks with cardinality biased towards fewer observed features.'''
    unif = torch.rand(batch_size, num_features, device=device)
    ref = torch.rand(batch_size, 1, device=device).pow(0.25)
    return (unif > ref).float()

class StudentEmbedder(nn.Module):
    '''
    Student model with separate classification and retrieval heads.
    
    Architecture:
    - Trunk h = MLP(x_tilde)
    - Classifier head: z = W_c h
    - Retrieval head: e = Proj(h), then normalize e/||e||
    '''
    def __init__(self, d_in, d_out, n_hidden=128, n_embed=64, dropout=0.3):
        super().__init__()
        self.d_in = d_in
        
        # Learnable parameters for missing values
        self.e_miss = nn.Parameter(torch.zeros(1, d_in))
        nn.init.normal_(self.e_miss, std=0.01)
        
        # MLP Trunk
        self.lin1 = nn.Linear(d_in, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        
        # Classifier Head
        self.classifier_head = nn.Linear(n_hidden, d_out)
        
        # Retrieval Head
        self.retrieval_head = nn.Linear(n_hidden, n_embed)
        
        self.dropout = dropout

    def forward(self, x, m):
        '''
        Returns:
            logits: Classification logits
            embeddings: Normalized embeddings for retrieval
        '''
        # Apply feature-wise missing embeddings
        x_tilde = x * m + (1 - m) * self.e_miss
        
        # Trunk
        h = F.relu(self.lin1(x_tilde))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.lin2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Heads
        logits = self.classifier_head(h)
        
        embeddings = self.retrieval_head(h)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return logits, embeddings

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        denominator = mask.sum(1)
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denominator

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class DistillationTrainer:
    '''
    Trainer for distilling a teacher model (Tree or MLP) into a StudentModel.
    '''
    def __init__(self, student_model, teacher_model, device='mps'):
        self.student = student_model
        self.teacher = teacher_model
        self.device = device
        self.student.to(device)
        

    def get_teacher_logits(self, x_full):
        '''
        Get logits from the teacher model on full features.
        '''
        x_np = x_full.cpu().numpy()
        
        if hasattr(self.teacher, "predict_proba"):
            probs = self.teacher.predict_proba(x_np)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            logits_np = np.log(probs)
            return torch.tensor(logits_np, device=self.device, dtype=torch.float32)
        else:
            preds = self.teacher.predict(x_np)
            return torch.tensor(preds, device=self.device, dtype=torch.float32).unsqueeze(1)

    def fit(self,
            train_loader,
            val_loader,
            n_epochs=50,
            lr=1e-3,
            temp=2.0,
            alpha=0.5,
            beta=0.1, # Weight for embedding loss
            gamma=1.0, # Bandwidth for consistency weight
            p_full=0.3,
            task_type='classification',
            verbose=True,
            contrast_temp=0.07):
        '''
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation.
            n_epochs: Number of epochs.
            lr: Learning rate.
            temp: Temperature for distillation.
            alpha: Weight for CE loss (1-alpha for KD loss).
            beta: Weight for Posterior-Consistency Loss.
            gamma: Bandwidth for consistency weight.
            p_full: Probability of using full mask (all observed) in a batch.
            task_type: 'classification' or 'regression'.
        '''
        optimizer = optim.Adam(self.student.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        # Loss functions

        ce_loss_fn = nn.CrossEntropyLoss()
        kl_div_fn = nn.KLDivLoss(reduction='batchmean')
        supcon_loss_fn = SupConLoss(temperature=contrast_temp)

            
        best_val_metric = -float('inf')
        best_model_state = None
        
        is_embedder = isinstance(self.student, StudentEmbedder)
        
        for epoch in range(n_epochs):
            self.student.train()
            total_loss = 0
            
            for batch in train_loader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch[0], batch[-1]
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                batch_size, n_features = x.shape
                
                # 1. Sample two masks
                # m1: sometimes full, sometimes random
                if np.random.rand() < p_full:
                    m1 = torch.ones(batch_size, n_features, device=self.device)
                else:
                    m1 = generate_uniform_mask(batch_size, n_features, device=self.device)
                
                # m2: always random (for consistency check)
                m2 = generate_uniform_mask(batch_size, n_features, device=self.device)
                
                # 2. Student forward (Double Batch)
                # We process both masks in one go for efficiency if possible, or just sequentially.
                # Let's do concatenation to keep it vectorized.
                x_double = torch.cat([x, x], dim=0)
                m_double = torch.cat([m1, m2], dim=0)
                
                if is_embedder:
                    logits_double, embeddings_double = self.student(x_double, m_double)
                    # Split back
                    logits1, logits2 = torch.split(logits_double, batch_size, dim=0)
                    emb1, emb2 = torch.split(embeddings_double, batch_size, dim=0)
                else:
                    logits_double = self.student(x_double, m_double)
                    logits1, logits2 = torch.split(logits_double, batch_size, dim=0)
                    embeddings_double = None
                
                # 3. Teacher forward (always full features)
                # We only need teacher logits for the original x once
                teacher_logits = self.get_teacher_logits(x)
                
                # 4. Compute Task Loss (Distillation + CE)
                # We can compute loss on both views to maximize training signal

                # Hard label loss (average over both views)
                loss_ce = 0.5 * (ce_loss_fn(logits1, y) + ce_loss_fn(logits2, y))
                
                # Distillation loss (average over both views)
                p_s1 = F.log_softmax(logits1 / temp, dim=1)
                p_s2 = F.log_softmax(logits2 / temp, dim=1)
                p_t = F.softmax(teacher_logits / temp, dim=1)
                
                loss_kd = 0.5 * (kl_div_fn(p_s1, p_t) + kl_div_fn(p_s2, p_t)) * (temp ** 2)
                
                loss_task = alpha * loss_ce + (1 - alpha) * loss_kd
                
                loss = loss_task
                
                # 5. Embedding Loss (InfoNCE)
                if is_embedder and beta > 0:
                    # Stack embeddings: [batch_size, 2, n_embed]
                    features = torch.stack([emb1, emb2], dim=1)
                    loss_emb = supcon_loss_fn(features)
                    
                    loss += beta * loss_emb
                    
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation

            val_acc, val_f1, val_auc = self.evaluate(val_loader)
            val_metric = val_acc
            
            if verbose and (epoch + 1) % 5 == 0:
                if  is_embedder and beta > 0:
                    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | CE Loss: {loss_ce:.4f} | KD Loss: {loss_kd:.4f} | Embedding Loss: {loss_emb:.4f}| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUROC: {val_auc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | CE Loss: {loss_ce:.4f} | KD Loss: {loss_kd:.4f}| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUROC: {val_auc:.4f}")

            
            # Save best
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_model_state = deepcopy(self.student.state_dict())

                    
        # Restore best
        if best_model_state is not None:
            self.student.load_state_dict(best_model_state)
            
        return best_val_metric

    def evaluate(self, loader):
        self.student.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []

        
        is_embedder = isinstance(self.student, StudentEmbedder)
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch[0], batch[-1]
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Evaluate on FULL features for now
                m = torch.ones_like(x)
                
                if is_embedder:
                    logits, _ = self.student(x, m)
                else:
                    logits = self.student(x, m)
                

                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

                    
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)
        
        # Calculate metrics
        acc = (all_preds == all_targets).float().mean().item()
        
        # F1 and AUROC
        try:

            
            # F1
            f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average='macro')
            
            # AUROC
            if all_probs.shape[1] == 2:
                auc = roc_auc_score(all_targets.numpy(), all_probs[:, 1].numpy())
            else:
                auc = roc_auc_score(all_targets.numpy(), all_probs.numpy(), multi_class='ovr', average='macro')
        except ImportError:
            print("Warning: sklearn not found, returning 0 for F1/AUC")
            f1 = 0.0
            auc = 0.0
            
        return acc, f1, auc


def evaluate_model(model, dataloader, device, mask_percentage=0.0):
    '''
    Evaluates StudentEmbedder or MLP_tabular on a dataloader.
    
    Args:
        model: The model to evaluate.
        dataloader: Yields (x, x_shap, y).
        device: Device to run on.
        mask_percentage: Percentage of features to mask (0.0 to 1.0).
                         If > 0, random mask is generated.
    '''
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch[0], batch[-1]
            
            x = x.to(device)
            y = y.to(device)
            
            N, D = x.shape
            
            # Generate mask
            if mask_percentage > 0:
                k = int(round(mask_percentage * D))
                # Randomly mask k features (set mask to 0)
                # Start with all ones
                m = torch.ones_like(x)
                if k > 0:
                    # Randomly select k indices to set to 0
                    # We use rand().topk() to get random indices
                    _, indices = torch.rand(N, D, device=device).topk(k, dim=1)
                    m.scatter_(1, indices, 0.0)
            else:
                m = torch.ones_like(x)
            # print(x)
            # print(m)
            # Forward
            if hasattr(model, 'retrieval_head'): # StudentEmbedder
                logits, _ = model(x, m)
                
            else:
                # MLP_tabular
                combined = torch.cat([x * m, m], dim=1)
                logits = model(combined)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())
            
    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    acc = (all_preds == all_targets).float().mean().item()
    
    # F1 and AUROC

    from sklearn.metrics import f1_score, roc_auc_score
    
    # F1
    f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average='macro')
    
    # AUROC
    if all_probs.shape[1] == 2:
        auc = roc_auc_score(all_targets.numpy(), all_probs[:, 1].numpy())
    else:
        auc = roc_auc_score(all_targets.numpy(), all_probs.numpy(), multi_class='ovr', average='macro')
    print(f'Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')


    return acc, f1, auc
