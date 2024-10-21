import os
import torch
import json
from torch_geometric.datasets import Planetoid, Reddit, PPI
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit
from sklearn.metrics import f1_score
from train import train_and_test_model  # Import from train.py
from models import GCN_Model, GCN_JK_Model  # Import the model from models.py
import numpy as np
import wandb

# wandb.login()

# Create directories if they don't exist
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')
if not os.path.exists('results'):
    os.makedirs('results')

# Helper functions to save and load models and results
def save_model(model, model_name, dataset_name):
    model_path = f'trained_models/{dataset_name}_{model_name}.pt'
    torch.save(model.state_dict(), model_path)

def load_model(model, model_name, dataset_name):
    model_path = f'trained_models/{dataset_name}_{model_name}.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded model from {model_path}")
        return True
    return False

def save_results(results, dataset_name, model_name):
    results_path = f'results/{dataset_name}_{model_name}_results.json'
    
    # Convert non-serializable types to standard Python types
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            results[key] = value.item()  # Convert tensor to Python float
        elif isinstance(value, np.ndarray):
            results[key] = value.tolist()  # Convert ndarray to a Python list
    
    with open(results_path, 'w') as f:
        json.dump(results, f)

def load_results(dataset_name, model_name):
    results_path = f'results/{dataset_name}_{model_name}_results.json'
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
        name, data, device, model_name_template,
        use_jk=False, jk_mode='max',
        ):
    num_nodes, num_edges, num_features, num_classes, is_multilabel = get_dataset_stats(data)
    print(f"Dataset: {name}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}\n")

    # Prepare data loaders
    if name == 'PPI':
        # Load PPI dataset with predefined splits
        train_dataset = PPI(root='data/PPI', split='train')
        val_dataset = PPI(root='data/PPI', split='val')
        test_dataset = PPI(root='data/PPI', split='test')
        batch_size = 2  # Adjust based on your memory constraints

        # Create DataLoaders for PPI
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Apply RandomNodeSplit transform to create train/val/test masks
        transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
        data = transform(data)
        verify_masks(data)  # Verify masks do not overlap
        data = data.to(device)  # Move data to the correct device

        batch_size = 32 if name == 'Reddit' else 1

        # Now create the subgraphs for the training, validation, and test sets
        train_set = data.subgraph(data.train_mask)
        val_set = data.subgraph(data.val_mask)
        test_set = data.subgraph(data.test_mask)

        # Create DataLoaders with a single Data object
        # train_loader = DataLoader([train_set], batch_size=batch_size, shuffle=False)
        # val_loader = DataLoader([val_set], batch_size=batch_size, shuffle=False)
        # test_loader = DataLoader([test_set], batch_size=batch_size, shuffle=False)

        loader = DataLoader([data], batch_size=1, shuffle=False)
        train_loader = val_loader = test_loader = loader

    # Model hyperparameters
    hyperparameters = dict(
        # layers = 6 if name in ('PPI', 'Reddit') else 3,
        # layers = 6,
        layers=2 if name in ('Citeseer', 'Cora') else 6,
        epochs = 300,
        learning_rate = 0.005,  # 0.005 in Xu2018
        weight_decay = 0.0005,  # 0.0005 in Xu2018
        # weight_decay = 0.001,  # Increase regularization
        # dropout = 0.5
        # dropout = 0.5 if name == 'Reddit' else 0.3
        dropout = 0.5,  # 0.5 in Xu2018
        hidden_size = 16 if name in ('Citeseer', 'Cora') else 32,
    )
    model_name = model_name_template.format(
        layers=hyperparameters['layers'], epochs=hyperparameters['epochs'])
    if use_jk is True:
        model_name = f'JK_{model_name}'  # Add JK prefix to distinguish models

    # Load saved results if available
    saved_results = load_results(name, model_name)
    if saved_results:
        print(f"Loading saved results for {name}: {saved_results}")
        mean_acc = saved_results['mean_accuracy']
        std_acc = saved_results['std_accuracy']
        micro_f1 = saved_results['micro_f1_score']
    else:
        # Define the model function
        def model_function(layers, dropout):
            if use_jk is True:
                model = GCN_JK_Model(
                    hidden_size=hyperparameters['hidden_size'], layers=layers, jk_mode=jk_mode, output_size=num_classes).to(device)
            else:
                model = GCN_Model(
                    hidden_size=hyperparameters['hidden_size'], layers=layers, output_size=num_classes).to(device)
            if load_model(model, model_name, name):
                print(f"Loaded model for {name}")
            return model

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
        save_model(model_trained, model_name, name)

        # Evaluate the model using the extracted function
        all_preds, all_labels = eval_model(
            model_trained, test_loader, device, is_multilabel,
        )

        # Calculate evaluation metrics
        micro_f1 = f1_score(all_labels, all_preds, average='micro')

        # Save results
        results = {
            'mean_accuracy': mean_acc.item(),
            'std_accuracy': std_acc.item(),
            'micro_f1_score': micro_f1,
        }
        save_results(results, name, model_name)

    print(f"Mean accuracy for {name}: {mean_acc}, Standard deviation: {std_acc}")
    print(f"Micro F1 Score for {name}: {micro_f1}")
    for k, v in hyperparameters.items():
        print(k, v)
    print('jk_mode', jk_mode)

    print()


def eval_model(model, loader, device, is_multilabel):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            if is_multilabel:
                preds = torch.sigmoid(out[batch.test_mask])
                preds = (preds > 0.5).float()
                labels = batch.y[batch.test_mask].float()
            else:
                preds = out[batch.test_mask].argmax(dim=1)
                labels = batch.y[batch.test_mask]
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
cora_data = Planetoid(root='data/Cora', name='Cora')[0]
reddit_data = Reddit(root='data/Reddit')[0]
ppi_data = PPI(root='data/PPI')[0]

# Dictionary of datasets
datasets = {
    'Citeseer': citeseer_data,
    'Cora': cora_data,
    'Reddit': reddit_data,
    'PPI': ppi_data
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Process each dataset
# for name, data in datasets.items():
#     num_nodes, num_edges, num_features, num_classes, is_multilabel = get_dataset_stats(data)
#     print(f"Dataset: {name}")
#     print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}")
#     print('is_multilabel', is_multilabel)
#     print()

jk_mode = 'max'  # Choose 'cat', 'max', or 'lstm'



for name, data in datasets.items():
    if name == 'Reddit': continue  # memory issue?
    verify_masks(data)
    wandb.init(project="gcn_project", name=f"{name}_run")
    process_dataset(
        name, data, device, model_name_template="{layers}_layers_{epochs}_epochs",
        use_jk=True, jk_mode=jk_mode)
    wandb.finish()  # End the WandB run after processing each dataset