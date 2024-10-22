import torch
from torcheval.metrics import MulticlassAccuracy
import torch.optim.adam
import torch_geometric.data.data
from typing import Callable
from torch_geometric.transforms import RandomNodeSplit
import wandb
from torch import nn

def train_loop(train_dataloader: torch_geometric.data.data.Data,
               model,
               loss_fn: torch.nn.modules.loss.MSELoss,
               optimizer: torch.optim.Adam,
               epochs: int,
               wandb_iteration: int,
               wandb_toggle = False):
  model.train()
  
  for epoch in range(epochs):
    pred = model(train_dataloader)
    loss = loss_fn(pred, train_dataloader.y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    #if epoch % 100 == 0:
    #  loss, current = loss.item(), epoch
    #  print(f"loss: {loss:>7f}  [{current:>5d}/{epoch:>5d}]")

    if wandb_toggle:
      wandb.log({f'plot iteration {wandb_iteration}': {'epoch': epoch + 1, 'loss': loss}})

def train_and_validate(train_dataloader: torch_geometric.data.data.Data,
                       val_dataloader: torch_geometric.data.data.Data,
                       model_function: Callable[int, float],
                       loss_fn: torch.nn.modules.loss.MSELoss,
                       layers: int,
                       epochs: int,
                       learning_rate: float,
                       weight_decay: float,
                       dropout: float,
                       wandb_iteration: int,
                       wandb_toggle = False):
  
  model = model_function(layers, dropout)
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

  train_loop(train_dataloader, model, loss_fn, optimizer, epochs, wandb_iteration, wandb_toggle)

  model.eval()

  with torch.no_grad():
    pred = model(val_dataloader)
    metric = MulticlassAccuracy()
    metric.update(torch.round(pred.argmax(-1)), val_dataloader.y)
    MCAccuracy = metric.compute()

  return (model, MCAccuracy) 

def best_model(train_dataloader: torch_geometric.data.data.Data,
               val_dataloader: torch_geometric.data.data.Data,
               model_function: Callable[int, float],
               loss_fn: torch.nn.modules.loss.MSELoss,
               num_of_layers: int,
               epochs: int):
  learning_rate = 0.005
  weight_decay = 0.0005
  dropout = 0.5

  model_performances = []

  for layers in range(1, num_of_layers + 1):
    (model, MCAccuracy) = train_and_validate(train_dataloader, val_dataloader, model_function, loss_fn,
                                             layers, epochs, learning_rate, weight_decay, dropout, 0)

    model_performances.append({"model": model, "MCAccuracy": MCAccuracy, "num_layers": layers})

  best_performance = max(model_performances, key = lambda e: e["MCAccuracy"])

  return best_performance

def test_on_testset(test_dataloader: torch_geometric.data.data.Data,
                    model,
                    device: str):
  MCAccuracies = []

  model.eval()

  with torch.no_grad():
    for _ in range(3):
      transform = RandomNodeSplit(split = "train_rest", num_val = 0.33, num_test = 0.33)
      test_data_split = transform(test_dataloader).to(device)
      test_data_split.subgraph(test_data_split["train_mask"])

      pred = model(test_data_split)
      metric = MulticlassAccuracy()
      metric.update(pred.argmax(-1), test_data_split.y)
      MCAccuracy = metric.compute()

      MCAccuracies.append(MCAccuracy)

  MCAccuracies = torch.Tensor(MCAccuracies)

  return (torch.mean(MCAccuracies), torch.std(MCAccuracies))

def train_and_test_model(train_dataloader: torch_geometric.data.data.Data,
                         val_dataloader: torch_geometric.data.data.Data,
                         test_dataloder: torch_geometric.data.data.Data,
                         model,
                         layers: int,
                         epochs: int,
                         learning_rate: float,
                         weight_decay: float,
                         dropout: float,
                         device: str):
  loss_fn = nn.CrossEntropyLoss()
  
  (model_trained, _) = train_and_validate(train_dataloader, val_dataloader, model, loss_fn, layers, epochs, learning_rate, weight_decay, dropout, 0)

  return test_on_testset(test_dataloder, model_trained, device)