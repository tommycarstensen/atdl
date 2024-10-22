import torch
from torch import nn
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, GATConv, JumpingKnowledge, MLP

import torch.nn.functional as F
from torch_geometric.nn import BatchNorm

class GCN_Model(nn.Module):
  def __init__(self, hidden_size: int, layers: int, output_size: int):
    super().__init__()

    self.model = Sequential('x, edge_index', [
      (GCNConv(-1, hidden_size), 'x, edge_index -> x'), 
      ReLU(inplace=True)] +
      [e for tup in [((GCNConv(hidden_size, hidden_size), 'x, edge_index -> x'), 
      ReLU(inplace=True)) for _ in range(1, layers)] for e in tup]
      + [Linear(hidden_size, output_size),
    ])

  def forward(self, x, edge_index):
    pred = self.model(x, edge_index)
    return pred
  
class GAT_Model(nn.Module):
  def __init__(self, dropout: float, hidden_size: int, layers: int, output_size: int):
    super().__init__()

    self.model = Sequential('x, edge_index', [
      (GATConv(-1, hidden_size), f'x, edge_index -> x'), 
      ReLU(inplace=True),
      (Dropout(p = dropout), 'x -> x')] +
      [e for tup in [((GATConv(hidden_size, hidden_size), f'x, edge_index -> x'), 
      ReLU(inplace=True),
      (Dropout(p = dropout), 'x -> x')) for i in range(1, layers)] for e in tup]
      + [Linear(hidden_size, output_size),
    ])

  def forward(self, x, edge_index):
    pred = self.model(x, edge_index)
    return pred

class GCN_JK_Model(nn.Module):
  def __init__(self, hidden_size: int, layers: int, jk_mode: str, output_size: int):
    super().__init__()

    self.model = Sequential('x, edge_index', [
      (GCNConv(-1, hidden_size), 'x, edge_index -> x1'),
      ReLU(inplace=True)] +
      [e for tup in [((GCNConv(hidden_size, hidden_size), f'x{i}, edge_index -> x{i + 1}'), 
      ReLU(inplace=True)) for i in range(1, layers)] for e in tup]
      + [(lambda x: x, f'[{", ".join([f'x{i}' for i in range(1, layers + 1)])}]' + ' -> xs'),
      (JumpingKnowledge(jk_mode, hidden_size, num_layers = layers), 'xs -> x'),
      Linear(hidden_size, output_size) if jk_mode == "max" or jk_mode == "lstm" else Linear(hidden_size * layers, output_size),
    ])

  def forward(self, x, edge_index):
    pred = self.model(x, edge_index)
    return pred


class GCN_JK_Model_with_BatchNorm(nn.Module):
    def __init__(self, hidden_size: int, layers: int, jk_mode: str, output_size: int):
        super().__init__()
        self.layers = layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First GCN layer
        self.convs.append(GCNConv(-1, hidden_size))
        # Add Batch Normalization after the first layer
        self.norms.append(BatchNorm(hidden_size))
        
        # Additional GCN layers
        for _ in range(1, layers):
            # Subsequent GCN layers
            self.convs.append(GCNConv(hidden_size, hidden_size))
            # Add Batch Normalization after each layer
            self.norms.append(BatchNorm(hidden_size))
        
        # Jumping Knowledge layer
        self.jk = JumpingKnowledge(jk_mode, hidden_size, layers)
        
        # Fully connected layer
        if jk_mode == 'cat':
            # If concatenating, the input size is hidden_size * layers
            self.fc = Linear(hidden_size * layers, output_size)
        else:
            # For 'max' or 'lstm' modes
            self.fc = Linear(hidden_size, output_size)
        
    def forward(self, x, edge_index):
        xs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # GCN layer
            x = conv(x, edge_index)
            # Apply Batch Normalization
            x = norm(x)
            # Activation function
            x = F.relu(x)
            # Collect outputs for Jumping Knowledge
            xs.append(x)
        
        # Apply Jumping Knowledge mechanism
        x = self.jk(xs)
        # Final fully connected layer
        x = self.fc(x)
        return x


class GAT_JK_Model(nn.Module):
  def __init__(self, dropout: float, hidden_size: int, layers: int, jk_mode: str, output_size: int):
    super().__init__()

    self.model = Sequential('x, edge_index', [
      (GATConv(-1, hidden_size), 'x, edge_index -> x1'),
      ReLU(inplace=True),
      Dropout(p = dropout)] +
      [e for tup in [((GATConv(hidden_size, hidden_size), f'x{i}, edge_index -> x{i + 1}'), 
      ReLU(inplace=True), Dropout(p = dropout)) for i in range(1, layers)] for e in tup]
      + [(lambda x: x, f'[{", ".join([f'x{i}' for i in range(1, layers + 1)])}]' + ' -> xs'),
      (JumpingKnowledge(jk_mode, hidden_size, num_layers = layers), 'xs -> x'),
      Linear(hidden_size, output_size) if jk_mode == "max" or jk_mode == "lstm" else Linear(hidden_size * layers, output_size),
    ])

  def forward(self, x, edge_index):
    pred = self.model(x, edge_index)
    return pred
