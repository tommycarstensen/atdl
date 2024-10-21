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


def model_function(layers, dropout):
    return GCN_Model(hidden_size=64, layers=layers, output_size=num_classes).to(device)


def process_dataset(name, data, device, model_name_template):
    num_nodes, num_edges, num_features, num_classes, is_multilabel = get_dataset_stats(data)
    print(f"Dataset: {name}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}\n")

    # Split into training, validation, and test sets
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    dataset_split = transform(data)
    dataset_split = dataset_split.to(device)  # Move data to the correct device

    # Debugging statement to check the shape of x
    print(f"Shape of x: {dataset_split.x.shape}")

    # Use Data objects directly
    train_data = dataset_split
    val_data = dataset_split
    test_data = dataset_split

    # # Set batch size based on the available memory or dataset size
    # batch_size = 32 if name == 'Reddit' else None  # Example batch sizes

    # # Use DataLoader to create mini-batches for training, validation, and test sets
    # train_loader = DataLoader([dataset_split], batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader([dataset_split], batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader([dataset_split], batch_size=batch_size, shuffle=False)

    # layers = 3
    # epochs = 100
    # learning_rate = 0.001
    # weight_decay = 0.0005
    # dropout = 0.5
    layers = 2
    epochs = 200
    learning_rate = 0.01
    weight_decay = 0.0005
    dropout = 0.5
    model_name = model_name_template.format(layers=layers, epochs=epochs)

    saved_results = load_results(name, model_name)
    if saved_results:
        print(f"Loading saved results for {name}: {saved_results}")
        mean_acc = saved_results['mean_accuracy']
        std_acc = saved_results['std_accuracy']
        micro_f1 = saved_results['micro_f1_score']
        macro_f1 = saved_results['macro_f1_score']
        weighted_f1 = saved_results['weighted_f1_score']
        f1 = saved_results['f1_score']
    else:
        # def model_function(layers, dropout):
        #     model = GCN_Model(
        #         hidden_size=64, layers=layers, output_size=num_classes).to(device)
        #     if load_model(model, model_name, name):
        #         print(f"Loaded model for {name}")
        #     return model
        def model_function(layers, dropout):
            jk_mode = 'cat'  # Choose 'cat', 'max', or 'lstm'
            model = GCN_JK_Model(
                hidden_size=64, layers=layers, jk_mode=jk_mode, output_size=num_classes).to(device)
            if load_model(model, model_name, name):
                print(f"Loaded model for {name}")
            return model

        model_trained, mean_acc, std_acc = train_and_test_model(
            train_data, val_data, test_data,
            # train_loader, val_loader, test_loader,
            model_function, layers, epochs, learning_rate, weight_decay, dropout, device,
            is_multilabel=is_multilabel,
            
        )
        save_model(model_trained, model_name, name)

        # Calculate F1 Score (Micro F1)
        pred = model_trained(test_data.x, test_data.edge_index)
        # pred = model_trained(dataset_split.x, dataset_split.edge_index)
        if is_multilabel:
            preds = torch.sigmoid(pred).cpu()
            preds = (preds > 0.5).float()
            labels = test_data.y.cpu()
            micro_f1 = f1_score(labels, preds, average='micro')
            macro_f1 = f1_score(labels, preds, average='macro')
            weighted_f1 = f1_score(labels, preds, average='weighted')
            f1 = f1_score(labels, preds, average=None)
        else:
            preds = pred.argmax(dim=-1).cpu()
            labels = test_data.y.cpu()
            micro_f1 = f1_score(labels, preds, average='micro')
            macro_f1 = f1_score(labels, preds, average='macro')
            weighted_f1 = f1_score(labels, preds, average='weighted')
            f1 = f1_score(labels, preds, average=None)

        results = {
            'mean_accuracy': mean_acc.item(),
            'std_accuracy': std_acc.item(),
            'micro_f1_score': micro_f1,
            'macro_f1_score': macro_f1,
            'weighted_f1_score': weighted_f1,
            'f1_score': f1,
        }
        save_results(results, name, model_name)

    print(f"Mean accuracy for {name}: {mean_acc}, Standard deviation: {std_acc}")
    print(f"Micro F1 Score for {name}: {micro_f1}\n")

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

# Process each dataset
for name, data in datasets.items():
    num_nodes, num_edges, num_features, num_classes, is_multilabel = get_dataset_stats(data)
    print(f"Dataset: {name}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}")
    print('is_multilabel', is_multilabel)
    print()

for name, data in datasets.items():
    # if name == 'Reddit': continue  # memory issue?
    process_dataset(name, data, device, model_name_template="GCN_{layers}_layers_{epochs}_epochs")