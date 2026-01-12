import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, TensorDataset
from collections import Counter

        

def count_labels(subset):
    labels = [subset.dataset.tensors[1][i].item() for i in subset.indices]  # Extract labels from subset
    return Counter(labels)

def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0, return_indices = False, tree_return=False, if_metabric=False, if_ckd=False, dataset_name=None):
    '''
    Split dataset into train, val, test.
    
    Args:
      dataset: PyTorch dataset object.
      val_portion: percentage of samples for validation.
      test_portion: percentage of samples for testing.
      random_state: random seed.
    '''
    # Shuffle sample indices.
    print(f'features: {len(dataset.features)}')

        # Try to load pre-computed splits
    loaded_splits = False
    if dataset_name:
        split_file = f'datasets/{dataset_name}_data_splits.npz'
        if os.path.exists(split_file):
            print(f"Loading pre-computed splits from {split_file}")
            try:
                data = np.load(split_file)
                train_inds = data['train_inds']
                val_inds = data['val_inds']
                test_inds = data['test_inds']
                loaded_splits = True
            except Exception as e:
                print(f"Failed to load splits from {split_file}: {e}")
        else:
            print(f"No pre-computed splits found at {split_file}")

    if not loaded_splits:  
        rng = np.random.default_rng(random_state)
        inds = np.arange(len(dataset))
        rng.shuffle(inds)

        n_val = int(val_portion * len(dataset))
        n_test = int(test_portion * len(dataset))
        # test_inds = inds[:n_test]
        # val_inds = inds[n_test:(n_test + n_val)]
        
        
        val_inds = inds[:n_val]
        test_inds = inds[n_val:(n_test + n_val)]

        
        train_inds = inds[(n_test+ n_val):] #  + n_val
        

    
    if return_indices:
        return train_inds, val_inds, test_inds


    
    #print(train_inds)
    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)

    if tree_return==True:
        train_x = dataset.tensors[0][train_inds,:]
        train_y = dataset.tensors[2][train_inds]

        val_x = dataset.tensors[0][val_inds,:]
        val_y = dataset.tensors[2][val_inds]

        test_x = dataset.tensors[0][test_inds,:]
        test_y = dataset.tensors[2][test_inds]

        return train_x,  train_y, val_x, val_y, test_x, test_y # train_x_shap is required for AACO
    else:
        return train_dataset, val_dataset, test_dataset#, val_inds

    
def load_ctgs(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('datasets/aids.csv')

    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.

    # aids_v0_tree_depth2_lr005_weighted
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) 
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset

def load_income(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('datasets/income.csv')

    if features is None:
        features = np.array([f for f in df.columns if f not in ['income']])
    else:
        assert 'income' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.

    # aids_v0_tree_depth2_lr005_weighted
    x = np.array(df.drop(['income'], axis=1)[features]).astype('float32')
    y = np.array(df['income']).astype('int64')
    # Create dataset object.

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) 
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset



def load_german_credit(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('datasets/german_credit.csv')
    # Set features.
    df['Outcome'] = df['label']
    df.drop(columns=['label'], inplace=True)
    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) 
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset

def load_student(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('datasets/student.csv')
    # Set features.
    df['Outcome'] = df['grade']
    df.drop(columns=['grade'], inplace=True)
    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.
       
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) 
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset

def load_wine(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('datasets/wine.csv')
    # Set features.
    df['Outcome'] = df['score']
    df.drop(columns=['score'], inplace=True)
    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.
       
    
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) 
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


def load_fraud(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('datasets/fraud_encode.csv')
    # Set features.
    df['Outcome'] = df['fraud_label']
    df.drop(columns=['fraud_label'], inplace=True)
    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.
       
    
    
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) 
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset



# Transform registry for easy access
DATASET_FUNCTIONS = {
    'ctgs': load_ctgs,
    'german_credit': load_german_credit,
    'income': load_income,
    'student': load_student,
    'wine': load_wine,
    'fraud': load_fraud,
}


def get_dataset(dataset_name, **kwargs):
    """
    Main function to get transforms for any dataset
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'imagenette', 'bloodmnist')
        **kwargs: Additional arguments passed to the specific transform function
    
    Returns:
        tuple: (pretrain_loader, trainloader, valloader, testloader)
    """
    if dataset_name not in DATASET_FUNCTIONS:
        available = list(DATASET_FUNCTIONS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
    
    return DATASET_FUNCTIONS[dataset_name](**kwargs)


