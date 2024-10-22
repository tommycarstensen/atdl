import torch
from torch import nn
from torch.nn import Linear, ReLU, Dropout, ModuleList
from torch_geometric.nn import Sequential, GCNConv, GATConv, JumpingKnowledge
from torch.nn.functional import log_softmax

class GCN_Model(nn.Module):
  def __init__(self, hidden_size: int, layers: int, output_size: int):
    super().__init__()

    module_list = []

    module_list.append((GCNConv(-1, hidden_size), 'x, edge_index -> x1'))
    module_list.append((ReLU(inplace = True), 'x1 -> x1'))

    for i in range(1, layers):
      conv = (GCNConv(hidden_size, hidden_size), f'x{i}, edge_index -> x{i+1}')
      relu = (ReLU(inplace = True), f'x{i+1} -> x{i+1}')
      module_list.extend([conv, relu])

    module_list.append((Linear(hidden_size, output_size), f'x{layers} -> x_out'))

    self.model = Sequential('x, edge_index', module_list)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    pred = self.model(x, edge_index)
    return pred
  
class GAT_Model(nn.Module):
  def __init__(self, dropout: float, hidden_size: int, layers: int, output_size: int):
    super().__init__()

    module_list = []

    module_list.append((GATConv(-1, hidden_size), 'x, edge_index -> x1'))
    module_list.append((ReLU(inplace = True), 'x1 -> x1'))
    module_list.append((Dropout(p = dropout), 'x1 -> x1'))

    for i in range(1, layers):
      conv = (GATConv(hidden_size, hidden_size), f'x{i}, edge_index -> x{i+1}')
      relu = (ReLU(inplace = True), f'x{i+1} -> x{i+1}')
      drop = (Dropout(p = dropout), f'x{i+1} -> x{i+1}')
      module_list.extend([conv, relu, drop])
      
    module_list.append((Linear(hidden_size, output_size), f'x{layers} -> x_out'))

    self.model = Sequential('x, edge_index', module_list)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    pred = self.model(x, edge_index)
    return pred

class GCN_JK_Model(nn.Module):
  def __init__(self, hidden_size: int, layers: int, jk_mode: str, output_size: int):
    super().__init__()

    module_list = []

    module_list.append((GCNConv(-1, hidden_size), 'x, edge_index -> x1'))
    module_list.append((ReLU(inplace = True), 'x1 -> x1'))

    for i in range(1, layers):
      conv = (GCNConv(hidden_size, hidden_size), f'x{i}, edge_index -> x{i+1}')
      relu = (ReLU(inplace = True), f'x{i+1} -> x{i+1}')
      module_list.extend([conv, relu])

    xs = ', '.join([f'x{i+1}' for i in range(layers)])
    module_list.append((JumpingKnowledge(jk_mode, hidden_size, num_layers = layers), f'[{xs}] -> x_out'))

    if jk_mode in ["max", "lstm"]:
      module_list.append((Linear(hidden_size, output_size), 'x_out -> x_out'))
    else:
      module_list.append((Linear(hidden_size * layers, output_size), 'x_out -> x_out'))

    self.model = Sequential('x, edge_index', module_list)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    pred = self.model(x, edge_index)
    return pred

class GAT_JK_Model(nn.Module):
  def __init__(self, dropout: float, hidden_size: int, layers: int, jk_mode: str, output_size: int):
    super().__init__()

    module_list = []

    module_list.append((GATConv(-1, hidden_size), 'x, edge_index -> x1'))
    module_list.append((ReLU(inplace = True), 'x1 -> x1'))
    module_list.append((Dropout(p = dropout), 'x1 -> x1'))

    for i in range(1, layers):
      conv = (GATConv(hidden_size, hidden_size), f'x{i}, edge_index -> x{i+1}')
      relu = (ReLU(inplace = True), f'x{i+1} -> x{i+1}')
      drop = (Dropout(p = dropout), f'x{i+1} -> x{i+1}')
      module_list.extend([conv, relu, drop])

    xs = ', '.join([f'x{i+1}' for i in range(layers)])
    module_list.append((JumpingKnowledge(jk_mode, hidden_size, num_layers = layers), f'[{xs}] -> x_out'))

    if jk_mode in ["max", "lstm"]:
      module_list.append((Linear(hidden_size, output_size), 'x_out -> x_out'))
    else:
      module_list.append((Linear(hidden_size * layers, output_size), 'x_out -> x_out'))

    self.model = Sequential('x, edge_index', module_list)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    pred = self.model(x, edge_index)
    return pred