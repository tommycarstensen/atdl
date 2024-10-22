import os
import torch
import json
from torch_geometric.datasets import Planetoid, Reddit, PPI
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit
from sklearn.metrics import f1_score
from train import train_and_test_model  # Import from train.py
# from assignment2.atdl.models_v1 import GCN_Model, GCN_JK_Model  # Import the model from models.py
from models import GCN_Model, GAT_Model, GCN_JK_Model, GAT_JK_Model
import numpy as np
import wandb
import time

# wandb.login()


# Create directories if they don't exist
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')
if not os.path.exists('results'):
    os.makedirs('results')

# Helper functions to save and load models and results
def save_model(model, name_formatted):
    model_path = f'trained_models/{name_formatted}.pt'
    torch.save(model.state_dict(), model_path)

def load_model(model, name_formatted):
    model_path = f'trained_models/{name_formatted}.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded model from {model_path}")
        return True
    return False

def save_results(results, name_formatted):
    results_path = f'results/{name_formatted}_results.json'
    
    # Convert non-serializable types to standard Python types
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            results[key] = value.item()  # Convert tensor to Python float
        elif isinstance(value, np.ndarray):
            results[key] = value.tolist()  # Convert ndarray to a Python list
    
    with open(results_path, 'w') as f:
        json.dump(results, f)

def load_results(name_formatted):
    results_path = f'results/{name_formatted}_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

# Function to calculate dataset statistics (nodes, edges, features, classes)
def get_dataset_stats(data):
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1] // 2
    num_features = data.x.shape[1]
    if len(data.y.shape) > 1 and data.y.shape[1] > 1:
        # Multi-label classification
        num_classes = data.y.shape[1]
        is_multilabel = True
    else:
        num_classes = int(data.y.max().item() + 1)
        is_multilabel = False
    return num_nodes, num_edges, num_features, num_classes, is_multilabel



def process_dataset(
        dataset_name, data, model_class, model_name,
        layers,
        device, name_template,
        use_jk=False, jk_mode='max',
        ):
    num_nodes, num_edges, num_features, num_classes, is_multilabel = get_dataset_stats(data)
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}\n")

    # Model hyperparameters
    hyperparameters = dict(
        # layers = 6 if name in ('PPI', 'Reddit') else 3,
        # layers = 6,
        # layers=2 if dataset_name in ('Citeseer', 'Cora') else 6,
        layers=layers,
        epochs = 1000 if dataset_name in ('Citeseer', 'Cora') else 1000,
        learning_rate = 0.005 if dataset_name in ('Citeseer', 'Cora') else 0.001,  # 0.005 in Xu2018
        weight_decay = 0.0005 if dataset_name in ('Citeseer', 'Cora') else 0,  # 0.0005 in Xu2018
        # weight_decay = 0.001,  # Increase regularization
        # dropout = 0.5
        # dropout = 0.5 if name == 'Reddit' else 0.3
        # dropout = 0.5,  # 0.5 in Xu2018
        dropout = 0.5 if 'GAT' in model_name else 0.0,
        hidden_size = 16 if dataset_name in ('Citeseer', 'Cora') else 256,
    )

    name_formatted = name_template.format(
        model_name=model_name,
        dataset_name=dataset_name,
        layers=hyperparameters['layers'],
        epochs=hyperparameters['epochs'],
        jk_mode=jk_mode,
        )

    if dataset_name == 'PPI':
        batch_size = 1  # Adjust based on your memory constraints
    else:
        batch_size = 32 if dataset_name == 'Reddit' else 1

    # Load saved results if available
    saved_results = load_results(name_formatted)
    if saved_results:
        print(f"Loading saved results for {dataset_name}: {saved_results}")
        mean_acc = saved_results['mean_accuracy']
        std_acc = saved_results['std_accuracy']
        micro_f1 = saved_results['micro_f1_score']
    else:

        torch.manual_seed(42)

        # Prepare data loaders
        if dataset_name == 'PPI':
            # Load PPI dataset with predefined splits
            train_dataset = PPI(root='data/PPI', split='train')
            val_dataset = PPI(root='data/PPI', split='val')
            test_dataset = PPI(root='data/PPI', split='test')

            # Create DataLoaders for PPI
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            # Apply RandomNodeSplit transform to create train/val/test masks
            # "We split nodes in each graph into 60%, 20% and 20% for training, validation and testing."
            transform = RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
            data = transform(data)
            verify_masks(data)  # Verify masks do not overlap
            data = data.to(device)  # Move data to the correct device

            # # Now create the subgraphs for the training, validation, and test sets
            # train_set = data.subgraph(data.train_mask)
            # val_set = data.subgraph(data.val_mask)
            # test_set = data.subgraph(data.test_mask)

            # Create DataLoaders with a single Data object
            # train_loader = DataLoader([train_set], batch_size=batch_size, shuffle=False)
            # val_loader = DataLoader([val_set], batch_size=batch_size, shuffle=False)
            # test_loader = DataLoader([test_set], batch_size=batch_size, shuffle=False)

            loader = DataLoader([data], batch_size=1, shuffle=False)
            train_loader = val_loader = test_loader = loader

        # Define the model function
        def model_function(layers, dropout):
            model_kwargs = {
                'hidden_size': hyperparameters['hidden_size'],
                'layers': layers,
                'output_size': num_classes,
            }
            if 'GAT' in model_name:
                model_kwargs['dropout'] = dropout
            if use_jk:
                model_kwargs['jk_mode'] = jk_mode

            model_instance = model_class(**model_kwargs).to(device)
            if load_model(model_instance, name_formatted):
                print(f"Loaded model for {dataset_name}")
            return model_instance

        t1 = time.time()
        wandb.init(project="gcn_project", name=f"{dataset_name}_run")

        # Train and test the model
        model_trained, mean_acc, std_acc = train_and_test_model(
            train_loader, val_loader, test_loader,
            model_function,
            hyperparameters['layers'],
            hyperparameters['epochs'],
            hyperparameters['learning_rate'],
            hyperparameters['weight_decay'],
            hyperparameters['dropout'],
            device,
            is_multilabel=is_multilabel,
            wandb_toggle=True,
        )

        wandb.finish()  # End the WandB run after processing each dataset
        print('runtime', time.time() - t1)

        save_model(model_trained, name_formatted)

        # Evaluate the model using the extracted function
        all_preds, all_labels = eval_model(
            model_trained, test_loader, device, is_multilabel,
        )

        # Calculate evaluation metrics
        micro_f1 = f1_score(all_labels, all_preds, average='micro')

        # Save results
        results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'micro_f1_score': micro_f1,
        }
        save_results(results, name_formatted)

    print('dataset_name', dataset_name)
    print('model_name', model_name)
    print(f"Mean accuracy for {dataset_name}: {mean_acc}, Standard deviation: {std_acc}")
    print(f"Micro F1 Score for {dataset_name}: {micro_f1}")
    for k, v in hyperparameters.items():
        print(k, v)
    if use_jk is True:
        print('jk_mode', jk_mode)
    print('batch_size', batch_size)

    with open('results.txt', 'a') as file:
        print(
            dataset_name, model_name, jk_mode, layers,
            round(mean_acc.item() if isinstance(mean_acc, torch.Tensor) else mean_acc, 3),
            round(std_acc.item() if isinstance(std_acc, torch.Tensor) else std_acc, 3),
            file=file, sep='\t',
        )

    print()


def eval_model(model, loader, device, is_multilabel):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)

            # Determine test indices
            if hasattr(batch, 'test_mask'):
                test_indices = batch.test_mask
            else:
                # Use all nodes in the batch
                test_indices = torch.arange(batch.num_nodes, device=device)

            # Get predictions and labels
            if is_multilabel:
                preds = torch.sigmoid(out[test_indices])
                preds = (preds > 0.5).float()
                labels = batch.y[test_indices].float()
            else:
                preds = out[test_indices].argmax(dim=1)
                labels = batch.y[test_indices]

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels

def verify_masks(data):
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    assert not torch.any(train_mask & val_mask), "Train and validation masks overlap!"
    assert not torch.any(train_mask & test_mask), "Train and test masks overlap!"
    assert not torch.any(val_mask & test_mask), "Validation and test masks overlap!"

# Load datasets
citeseer_data = Planetoid(root='data/Citeseer', name='Citeseer')[0]
assert len(Planetoid(root='data/Citeseer', name='Citeseer')) == 1
cora_data = Planetoid(root='data/Cora', name='Cora')[0]
assert len(Planetoid(root='data/Cora', name='Cora')) == 1
reddit_data = Reddit(root='data/Reddit')[0]
assert len(Reddit(root='data/Reddit')) == 1
ppi_data = PPI(root='data/PPI')

# Dictionary of datasets
datasets = {
    'Citeseer': citeseer_data,
    'Cora': cora_data,
    'Reddit': reddit_data,
    'PPI': ppi_data
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # PPI stats
# for i, data in enumerate(PPI(root='data/PPI')):
#     num_nodes, num_edges, num_features, num_classes, is_multilabel = get_dataset_stats(data)
#     print(f"{i}, Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}")

# for name, data in datasets.items():
#     num_nodes, num_edges, num_features, num_classes, is_multilabel = get_dataset_stats(data)
#     print(f"Dataset: {name}")
#     print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}")
#     print('is_multilabel', is_multilabel)
#     print()

models = {
    'GCN': GCN_Model,
    'GAT': GAT_Model,
    'GCN_JK': GCN_JK_Model,
    'GAT_JK_Model': GAT_JK_Model,
}

for dataset_name, data in datasets.items():
    if dataset_name == 'Reddit': continue  # memory issue?
    if dataset_name != 'PPI':
        verify_masks(data)
    # for use_jk in (True, False):
    for model_name, model_class in models.items():
        if 'JK' in model_name:
            use_jk = True
            jk_modes = ['max', 'cat', 'lstm']
        else:
            use_jk = False
            jk_modes = ['']
        for jk_mode in jk_modes:
            if dataset_name not in ('Cora', 'Citeseer'):
                layers = 6
            else:
                if dataset_name == 'Citeseer':
                    if model_name in ('GCN', 'GAT'):
                        layers = 2
                    elif 'JK' in model_name:
                        if jk_mode in ('max', 'cat'):
                            layers = 1
                        else:
                            layers = 2
                if dataset_name == 'Cora':
                    if model_name == 'GCN':
                        layers = 2
                    elif model_name == 'GAT':
                        layers = 3
                    elif 'JK' in model_name:
                        if jk_mode in ('max', 'cat'):
                            layers = 6
                        else:
                            layers = 1
            process_dataset(
                dataset_name, data, model_class, model_name,
                layers,
                device,
                name_template="{dataset_name}_{model_name}_{jk_mode}_{layers}_layers_{epochs}_epochs",
                use_jk=use_jk, jk_mode=jk_mode)
