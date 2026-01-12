import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
import re
import random
from datetime import datetime
import json
import os
import concurrent.futures


from embedding import AFAVectorDatabase
from llm_agent import query_llm
from prompts_config import PROMPTS
from utils import feature_value_map
from student import StudentEmbedder

def parse_llm_response(response, available_features):
    """
    Parses the LLM response to extract a valid feature name.
    Expected format: "Decision: Acquire [feature_name]"
    """
    # Regex search for "Acquire [feature_name]"
    match = re.search(r"Decision: Acquire (.*?) now", response, re.IGNORECASE)
    if match:
        extracted_feature = match.group(1).strip()
        # Case-insensitive lookup
        for feat in available_features:
            if feat.lower() == extracted_feature.lower():
                return feat
        print("No valid feature found in LLM response.")
        return None
    else:
        print("Cant parse LLM response.")
        return None

def parse_llm_classification_response(response):
    """
    Parses the LLM response to extract a classification label.
    Expected format: "Diagnosis: [Label]" or similar.
    """
    # Regex search for "Diagnosis: [Label]"
    match = re.search(r"Decision: Diagnosis is .*?(\d+)", response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback: look for just a number
    match = re.search(r"\b(\d+)\b", response)
    if match:
        return int(match.group(1))
        
    return None

def run_prediction_pipeline(
    test_loader, api_key,
    afa_db: AFAVectorDatabase, 
    classifier: StudentEmbedder,
    mean: torch.Tensor,
    std: torch.Tensor,
    n_features_to_acquire: int = 5, 
    k_neighbors: int = 5,
    device: str = 'cpu',
    acquirer: str = 'gpt-5-mini',
    dataset_name: str = 'default',
    resume_from_file: str | None = None,
    anonymous_feature: bool = False,
    num_threads: int = 1,
    result_file: str = 'results.json',
    print_llm_response: bool = False,
    
):
    classifier.eval()
    classifier.to(device)
    
    # Ensure mean/std are on device
    mean = mean.to(device)
    std = std.to(device)
    
    all_preds = []
    all_labels = []
    
    feature_names = [str(f) for f in afa_db.features]
    
    if anonymous_feature:
        assert k_neighbors > 0, "Cannot use anonymous features with k_neighbors=0"
        display_feature_names = [f"Feature {i}" for i in range(len(feature_names))]
    else:
        display_feature_names = feature_names
        
    available_feats_str = ", ".join(display_feature_names)
    
    detailed_logs = []
    processed_indices = set()


    # Get prompts for the dataset
    if anonymous_feature:
        dataset_prompts = PROMPTS.get('anonymous')
    else:
        dataset_prompts = PROMPTS.get(dataset_name)
        
    acquisition_prompt_template = dataset_prompts['acquisition_prompt']
    final_prompt_template = dataset_prompts['final_prompt']
    
    # Optional k=0 prompts
    acquisition_prompt_k0 = dataset_prompts.get('acquisition_prompt_k0', acquisition_prompt_template)
    final_prompt_k0 = dataset_prompts.get('final_prompt_k0', final_prompt_template)
    
    # Prepare list of instances to process
    instances_to_process = []
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_size = x_batch.shape[0]
        for i in range(batch_size):
            instance_global_idx = batch_idx * batch_size + i
            if instance_global_idx not in processed_indices:
                instances_to_process.append({
                    'x_raw': x_batch[i],
                    'y_true': y_batch[i].item(),
                    'idx': instance_global_idx
                })

    print(f"Starting prediction pipeline on {len(test_loader.dataset)} instances ({len(instances_to_process)} remaining)...")
    print(f"Using {num_threads} threads.")

    def process_instance(inst_data):
        x_raw = inst_data['x_raw']
        y_true = inst_data['y_true']
        instance_global_idx = inst_data['idx']

        # Start with empty mask
        mask = torch.zeros_like(x_raw).to(device)
        acquired_indices = []
        acquisition_history = [] # Track sequence of feature names
        
        # Track trajectory of predictions and confidence
        pred_trajectory = []
        conf_trajectory = []
        logits_trajectory = []
        neighbor_label_trajectory = []
        neighbor_feature_trajectory = []
        
        # Track LLM conversation
        conversation_log = []
        
        # Pre-calculate normalized input for the instance
        x_norm = (x_raw - mean) / std
        
        # Active Feature Acquisition Loop
        for step in range(n_features_to_acquire):
            # 1. Run Prediction on Current Mask
            x_masked = x_norm * mask
            combined_input = torch.cat([x_masked, mask], dim=0).unsqueeze(0) # Add batch dim
            
            with torch.no_grad():
                if hasattr(classifier, 'retrieval_head'): # StudentEmbedder
                    # StudentEmbedder expects (x, m) with batch dim
                    x_in = x_norm.unsqueeze(0)
                    m_in = mask.unsqueeze(0)
                    logits, _ = classifier(x_in, m_in)
                else:
                    logits = classifier(combined_input)
                    
                probs = torch.softmax(logits, dim=1)
                current_pred_label = torch.argmax(probs, dim=1).item()
                current_confidence = probs[0, current_pred_label].item()
            
            # Record logits trajectory
            current_logits_dict = {i: round(probs[0, i].item(), 4) for i in range(probs.shape[1])}
            logits_trajectory.append(current_logits_dict)
            
            # Store current prediction and confidence
            pred_trajectory.append(current_pred_label)
            conf_trajectory.append(current_confidence)
            
            # 2. Construct Partial Instance Dictionary
            # Get current observed features
            observed_indices = torch.where(mask == 1)[0].cpu().numpy()
            observed_dict = {}
            
            # Use raw values for display
            for idx in observed_indices:
                feat_name = display_feature_names[idx]
                feat_val = x_raw[idx].item()
                
                # Map value if applicable
                if anonymous_feature:
                    mapped_val = feat_val
                else:
                    mapped_val = feature_value_map(feature_names[idx], feat_val, dataset_name)
                observed_dict[feat_name] = mapped_val if isinstance(mapped_val, str) else round(mapped_val, 4)
            
            # 3. Retrieve Context
            observed_dict_norm = {}
            for idx in observed_indices:
                feat_name = feature_names[idx]
                # Normalize using provided mean/std
                val_norm = (x_raw[idx] - mean[idx]) / std[idx]
                observed_dict_norm[feat_name] = val_norm.item()
            if k_neighbors == 0:
                retrieved_cases = []
            else:
                retrieved_cases = afa_db.retrieve_context(observed_dict_norm, k=k_neighbors)
            
            # Store neighbor labels and features
            current_step_neighbor_labels = []
            current_step_neighbor_features = []
            
            for rc in retrieved_cases:
                current_step_neighbor_labels.append(rc['case_label'])
                
                # Extract and normalize features for this neighbor
                # rc['full_features'] contains raw values
                vals = []
                for i, feat_name in enumerate(feature_names):
                    raw_val = rc['full_features'].get(feat_name, 0.0) # Safety get
                    norm_val = (raw_val - mean[i].item()) / std[i].item()
                    vals.append(norm_val)
                current_step_neighbor_features.append(vals)
                
            neighbor_label_trajectory.append(current_step_neighbor_labels)
            neighbor_feature_trajectory.append(current_step_neighbor_features)
            
            # 4. Construct Prompt
            # Format retrieved cases
            cases_str = ""
            for rc in retrieved_cases:
                full_feats_list = []
                for k, v in rc['full_features'].items():
                    if k in feature_names:
                        f_idx = feature_names.index(k)
                        # val_raw = v * std[f_idx].item() + mean[f_idx].item()
                        val_raw = v
                        
                        # Map value
                        if anonymous_feature:
                            mapped_val = val_raw
                        else:
                            mapped_val = feature_value_map(k, val_raw, dataset_name)
                        val_str = mapped_val if isinstance(mapped_val, str) else round(mapped_val, 2)
                        
                        full_feats_list.append(f"{str(display_feature_names[f_idx])}: {val_str}")
                    else:
                        full_feats_list.append(f"{str(k)}: {round(v, 2)}")
                full_feats_str = ", ".join(full_feats_list)
                
                suggested_feat = rc['suggested_feature']
                if anonymous_feature and suggested_feat in feature_names:
                        f_idx = feature_names.index(suggested_feat)
                        suggested_feat = display_feature_names[f_idx]
                        
                cases_str += f"- Case ID {rc['case_id']} (Label: {rc['case_label']}):\n    Most informative feature to acquire: {suggested_feat}\n    Full History: {full_feats_str}\n"

            # Filter available features (not yet acquired)
            current_available_features = [f for i, f in enumerate(display_feature_names) if mask[i] == 0]
            available_feats_str = ", ".join(current_available_features)
            
            if k_neighbors == 0:
                prompt = acquisition_prompt_k0.format(
                    observed_dict=observed_dict,
                    current_pred_label=current_pred_label,
                    current_confidence=current_confidence,
                    available_feats_str=available_feats_str
                )
            else:
                prompt = acquisition_prompt_template.format(
                    observed_dict=observed_dict,
                    current_pred_label=current_pred_label,
                    current_confidence=current_confidence,
                    cases_str=cases_str,
                    available_feats_str=available_feats_str
                )
            
            if anonymous_feature:
                prompt = re.sub(r"Feature explanation:.*?\n", "", prompt)
            
            if acquirer == 'random':
                next_feature = random.choice(current_available_features)

            elif acquirer == 'neighbors' or acquirer == 'neighbor':
                vote_scores = {}
                for rc in retrieved_cases:
                    suggested_feat = rc['suggested_feature']
                    if suggested_feat and suggested_feat in current_available_features:
                         # Voting score = similarity * entropy utility
                         # distance is 1 - similarity (lower is better), so we want similarity (higher is better)
                         # entropy utility is rc['feature_score']
                         sim = rc.get('similarity', 0.0) 
                         # Fallback if similarity missing but distance present? 
                         # But retrieve_context ensures 'similarity' is there.
                         
                         util = rc.get('feature_score', 0.0)
                         
                         score = sim * util
                         
                         if suggested_feat in vote_scores:
                             vote_scores[suggested_feat] += score
                         else:
                             vote_scores[suggested_feat] = score
                
                if vote_scores:
                    next_feature = max(vote_scores, key=vote_scores.get)
                else:
                    # Fallback to random if no votes or no valid suggestions
                    next_feature = random.choice(current_available_features)

            else:
                # 4. Query LLM
                if print_llm_response:
                    print("\nStep", step, " Prompt:", prompt)
                llm_response = query_llm(prompt,api_key=api_key,model=acquirer)
                if print_llm_response:
                    print("LLM Response:", llm_response)

                # Log conversation
                conversation_log.append({
                    "step": f"acquisition_step_{step}",
                    "prompt": prompt,
                    "response": llm_response
                })

                # 5. Parse Response
                next_feature = parse_llm_response(llm_response, display_feature_names)
            
            # 6. Update Mask
            if next_feature and next_feature in display_feature_names:
                feat_idx = display_feature_names.index(next_feature)
                if mask[feat_idx] == 0:
                    mask[feat_idx] = 1
                    acquired_indices.append(feat_idx)
                    acquisition_history.append(next_feature)
                else:
                    print(f'LLM suggested already acquired feature: {next_feature}. ')
                    pass
            else:
                print(f'Invalid response: {next_feature}. Skip.')
                pass
        
        # End of Acquisition Loop
        
        # Retrieval for final state
        observed_indices = torch.where(mask == 1)[0].cpu().numpy()
        observed_dict_norm = {}
        for idx in observed_indices:
            feat_name = feature_names[idx]
            # Normalize using provided mean/std
            val_norm = (x_raw[idx] - mean[idx]) / std[idx]
            observed_dict_norm[feat_name] = val_norm.item()
            
        if k_neighbors == 0:
            retrieved_cases = []
        else:
            retrieved_cases = afa_db.retrieve_context(observed_dict_norm, k=k_neighbors)
        
        # Store neighbor labels and features
        current_step_neighbor_labels = []
        current_step_neighbor_features = []
        
        for rc in retrieved_cases:
            current_step_neighbor_labels.append(rc['case_label'])
            
            # Extract and normalize features for this neighbor
            vals = []
            for i, feat_name in enumerate(feature_names):
                raw_val = rc['full_features'].get(feat_name, 0.0)
                norm_val = (raw_val - mean[i].item()) / std[i].item()
                vals.append(norm_val)
            current_step_neighbor_features.append(vals)
            
        neighbor_label_trajectory.append(current_step_neighbor_labels)
        neighbor_feature_trajectory.append(current_step_neighbor_features)

        # 7. Final Prediction
        # Normalize inputs for MLP
        x_norm = (x_raw - mean) / std
        
        # Prepare input for MLP: [x*mask, mask]
        x_masked = x_norm * mask
        combined_input = torch.cat([x_masked, mask], dim=0).unsqueeze(0) # Add batch dim
        
        # Always run MLP first to get context
        with torch.no_grad():
            if hasattr(classifier, 'retrieval_head'): # StudentEmbedder
                # StudentEmbedder expects (x, m) with batch dim
                x_in = x_norm.unsqueeze(0)
                m_in = mask.unsqueeze(0)

                logits, _ = classifier(x_in, m_in)
                
            probs = torch.softmax(logits, dim=1)
            nn_pred_label = torch.argmax(probs, dim=1).item()
            nn_confidence = probs[0, nn_pred_label].item()
            pred_trajectory.append(nn_pred_label)
            conf_trajectory.append(nn_confidence)

            # Record final logits
            current_logits_dict = {i: round(probs[0, i].item(), 4) for i in range(probs.shape[1])}
            logits_trajectory.append(current_logits_dict)
        
        pred_label = nn_pred_label
        pred_confidence = nn_confidence
        if probs.shape[1] == 2:
            pred_prob = probs[0, 1].item()
        else:
            pred_prob = probs[0].cpu().numpy()
        
        


        # print(f'Predicted Label: {pred_label}, True Label: {y_true}')
        
        log_entry = {
            "instance_idx": instance_global_idx,
            "gt_label": y_true,
            "prediction": pred_label,
            "pred_prob": float(pred_prob) if isinstance(pred_prob, (float, int)) else pred_prob.tolist(),
            "confidence": pred_confidence,
            "nn_prediction": nn_pred_label,
            "nn_confidence": nn_confidence,
            "acquired_features": acquisition_history,
            "prediction_trajectory": pred_trajectory,
            "confidence_trajectory": conf_trajectory,
            "logits_trajectory": logits_trajectory,
            "retrieved_neighbors_label_trajectory": neighbor_label_trajectory,
            "retrieved_neighbors_feature_trajectory": neighbor_feature_trajectory,
            "instance_normalized_features": x_norm.cpu().tolist(),
            "conversation_log": conversation_log
        }
        
        return log_entry, pred_prob, y_true

    # Run processing
    if num_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_instance, inst): inst for inst in instances_to_process}
            
            with tqdm(total=len(instances_to_process), desc="Processing Instances") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        log_entry, pred_prob, y_true = future.result()
                        detailed_logs.append(log_entry)
                        all_preds.append(pred_prob)
                        all_labels.append(y_true)
                        
                        if resume_from_file:
                            with open(resume_from_file, 'a') as f:
                                f.write(json.dumps(log_entry) + "\n")
                                
                    except Exception as e:
                        print(f"Instance failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    pbar.update(1)
    else:
        # Sequential
        with tqdm(total=len(instances_to_process), desc="Processing Instances") as pbar:
            for inst in instances_to_process:
                try:
                    log_entry, pred_prob, y_true = process_instance(inst)
                    detailed_logs.append(log_entry)
                    all_preds.append(pred_prob)
                    all_labels.append(y_true)
                    
                    if resume_from_file:
                        with open(resume_from_file, 'a') as f:
                            f.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    print(f"Instance failed: {e}")
                    import traceback
                    traceback.print_exc()
                pbar.update(1)
            
    # Calculate Metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Check if binary or multiclass based on preds shape
    # If all_preds is 1D (N,), it's binary probabilities for class 1
    # If all_preds is 2D (N, C), it's multiclass probabilities
    
    if all_preds.ndim == 1:
        # Binary
        # Handle case with only one class in labels for AUC (raises error)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.0 # Undefined if only one class present
            
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    else:
        # Multiclass
        # all_preds should be (N, C)
        # We need to stack them properly if they are list of arrays
        if isinstance(all_preds[0], np.ndarray) or all_preds.ndim == 2:
            if isinstance(all_preds, list):
                all_preds = np.vstack(all_preds)
            
            try:
                auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
            except ValueError:
                auc = 0.0
                
            preds_cls = np.argmax(all_preds, axis=1)
            acc = accuracy_score(all_labels, preds_cls)
            f1 = f1_score(all_labels, preds_cls, average='weighted')
        else:
             # Fallback if something went wrong
             auc = 0.0
             acc = 0.0
             f1 = 0.0
 

    
    results = {
        "accuracy": acc,
        "auroc": auc,
        "f1_score": f1,
        "detailed_logs": detailed_logs
    }
    
    print(f"Prediction Pipeline Completed.")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUROC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    

    # Save sorted detailed logs with conversations to a final file
    # Save sorted detailed logs to final files

    if detailed_logs and dataset_name:

        sorted_logs = sorted(detailed_logs, key=lambda x: x['instance_idx'])
        
        # Save sorted detailed logs to final files

        # Create output directories if they don't exist
        os.makedirs('results', exist_ok=True)
        os.makedirs('llm_convo_results', exist_ok=True)

        # 1. Save Full LLM Conversation Logs
        if acquirer not in ['neighbor','neighbors','random']:
            llm_log_file = os.path.join('llm_convo_results', f'llm_convo_{os.path.basename(result_file)}')
            with open(llm_log_file, 'w') as f:
                json.dump(sorted_logs, f, indent=4)
            print(f"Saved detailed LLM conversation logs to {llm_log_file}")
        
        # 2. Save Prediction Logs (without conversation logs)
        prediction_logs = []
        for log in sorted_logs:
            # Create a copy to avoid modifying the original if it were to be used later
            log_copy = log.copy()
            if 'conversation_log' in log_copy:
                del log_copy['conversation_log']
            prediction_logs.append(log_copy)
            
        pred_log_file = os.path.join('results', os.path.basename(result_file))
        
        with open(pred_log_file, 'w') as f:
            json.dump(prediction_logs, f, indent=4)
        print(f"Saved prediction logs (metrics only) to {pred_log_file}")
            


    return results
