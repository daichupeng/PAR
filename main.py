import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import os
import argparse
from datetime import datetime

from embedding import *
from student import *
from prediction import *
from tree_training import *
from tabular_dataset import *




def main(dataset_name,
    api_key,
    device='cuda',
    test_method='gpt-5-mini',    
    k_neighbors = 5,
    anonymous_feature=False,
    print_llm_results=False,
    num_threads=32):

    device = device

    dataset_name = dataset_name
    dataset = get_dataset(dataset_name)
    d_in = dataset.input_size
    d_out = dataset.output_size
    NUM_WORKERS = 8

    # split FIRST on the original (unnormalized) dataset
    train_idx, val_idx, test_idx = data_split(
        dataset, return_indices=True, test_portion=0.1,dataset_name=dataset_name,val_portion=0.1
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)
    test_dataset  = Subset(dataset, test_idx)


    # normalize each split independently and return stats
    def normalize_subset(subset, mean=None, std=None):
        X = subset.dataset.tensors[0][subset.indices]
        c = subset.dataset.tensors[-1][subset.indices]

        if mean is None:
            mean = X.mean(dim=0)
            std  = torch.clamp(X.std(dim=0), min=1e-3)

        normalized = torch.utils.data.TensorDataset(
            (X - mean) / std, c
        )
        return normalized, mean, std


    normalized_train_dataset, train_mean, train_std = normalize_subset(train_dataset)
    normalized_val_dataset,   val_mean,   val_std   = normalize_subset(val_dataset,train_mean, train_std)
    normalized_test_dataset,  test_mean,  test_std  = normalize_subset(test_dataset,train_mean, train_std)


    normalized_trainloader = torch.utils.data.DataLoader(
        normalized_train_dataset, batch_size=128, shuffle=True, num_workers=NUM_WORKERS
    )
    normalized_valloader = torch.utils.data.DataLoader(
        normalized_val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS
    )
    normalized_testloader = torch.utils.data.DataLoader(
        normalized_test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=NUM_WORKERS
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS
    )

    print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

    number_of_actions = d_in

    teacher = TreeMaskingPretrainer(
        model_type='catboost',
    )

    # print('beginning pre-training...')
    teacher.fit(
        normalized_trainloader,
        normalized_valloader,
        num_classes=2,
        verbose=True)
    print('Teacher training done.')

    # evaluate_tree_model(pretrain,normalized_valloader)
    # evaluate_tree_model(pretrain,normalized_testloader)

    student = StudentEmbedder(d_in=d_in,d_out=2,n_hidden=512,n_embed=128)
    trainer = DistillationTrainer(student, teacher, device=device)
    trainer.fit(normalized_trainloader, normalized_valloader, n_epochs=10, verbose=True, p_full=0.2,beta=0.1,lr=0.001,alpha=0.5)
    # evaluate_model(student,normalized_valloader,device=device)
    # evaluate_model(student,normalized_testloader,device=device)
    PATH = f'{dataset_name}.pth'
    torch.save(student.state_dict(), PATH)
    print('Student training done.')

    afa_db = AFAVectorDatabase(model=student,train_loader=trainloader,features=dataset.features)
    afa_db.build_index(additional_loader=valloader,mean=train_mean,std=train_std)


    test_method = test_method
    k_neighbors = 5
    anonymous_feature=False

    timestring = datetime.strftime(datetime.now(),format='%Y%m%d_%H%M%S')

    if anonymous_feature:
        an = f'_anon_{timestring}.json'
    else:
        an = f'_{timestring}.json'

    result_file = f'{dataset_name}_{test_method}_{k_neighbors}neib{an}'
    result = run_prediction_pipeline(
        testloader,
        afa_db=afa_db,
        classifier=student,
        mean=train_mean,
        std=train_std,
        n_features_to_acquire=len(dataset.features),
        device=device,
        k_neighbors=k_neighbors,
        acquirer=test_method,
        result_file = f'results/{result_file}',
        print_llm_response = print_llm_results,
        dataset_name=dataset_name,anonymous_feature=anonymous_feature,num_threads=num_threads,api_key=api_key)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AFA prediction pipeline.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--api_key', type=str, default=os.environ.get('openai_api_key'), help='API Key for the test method')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (default: cuda)')
    parser.add_argument('--test_method', type=str, default='gpt-5-mini', help='Test method (default: gpt-5-mini)')
    parser.add_argument('--precedents', type=int, default=5, help='Number of neighbors (default: 5)')
    parser.add_argument('--anonymous_feature', action='store_true', help='Use anonymous features')
    parser.add_argument('--print_llm_results', action='store_true', help='Print LLM results')
    parser.add_argument('--num_threads', type=int, default=32, help='Number of threads (default: 32)')

    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        api_key=args.api_key,
        device=args.device,
        test_method=args.test_method,
        k_neighbors=args.precedents,
        anonymous_feature=args.anonymous_feature,
        print_llm_results=args.print_llm_results,
        num_threads=args.num_threads
    )