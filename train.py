import torch
from torcheval.metrics import MulticlassAccuracy
import torch.optim.adam
import torch_geometric.data.data
from typing import Callable
from torch_geometric.transforms import RandomNodeSplit
import wandb
from torch import nn

def train_loop(
    train_dataloader: torch_geometric.data.data.Data,
    model,
    loss_fn: torch.nn.modules.loss.MSELoss,
    optimizer: torch.optim.Adam,
    epochs: int,
    wandb_iteration: int,
    wandb_toggle=False,
    is_multilabel=False,
):
    model.train()

    for epoch in range(epochs):
        pred = model(train_dataloader.x, train_dataloader.edge_index)
        if is_multilabel:
            loss = loss_fn(pred, train_dataloader.y.float())
        else:
            loss = loss_fn(pred, train_dataloader.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if wandb_toggle:
            wandb.log({f'plot iteration {wandb_iteration}': {'epoch': epoch + 1, 'loss': loss}})


def train_loop_with_early_stopping(
    train_dataloader: torch_geometric.data.data.Data,
    val_dataloader: torch_geometric.data.data.Data,
    model,
    loss_fn: torch.nn.modules.loss.MSELoss,
    optimizer: torch.optim.Adam,
    epochs: int,
    patience: int = 10,  # Patience for early stopping
    wandb_iteration: int = 0,
    wandb_toggle=False,
    is_multilabel=False,
):
    model.train()
    best_val_loss = float('inf')  # Track the best validation loss
    patience_counter = 0  # To keep track of how many epochs since the last improvement

    for epoch in range(epochs):
        # Training phase
        pred = model(train_dataloader.x, train_dataloader.edge_index)
        if is_multilabel:
            loss = loss_fn(pred, train_dataloader.y.float())
        else:
            loss = loss_fn(pred, train_dataloader.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(val_dataloader.x, val_dataloader.edge_index)
            if is_multilabel:
                val_loss = loss_fn(val_pred, val_dataloader.y.float())
            else:
                val_loss = loss_fn(val_pred, val_dataloader.y)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss}")
            break

        # Log to Weights and Biases if enabled
        if wandb_toggle:
            wandb.log({f'plot iteration {wandb_iteration}': {'epoch': epoch + 1, 'train_loss': loss, 'val_loss': val_loss}})
        
        model.train()  # Set the model back to training mode after validation


def train_and_validate(
    train_dataloader, val_dataloader, model_function, loss_fn, layers, epochs, learning_rate, weight_decay, dropout, wandb_iteration, is_multilabel=False, wandb_toggle=False,
    ):
    model = model_function(layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loop_with_early_stopping(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, wandb_iteration, wandb_toggle, is_multilabel=is_multilabel)

    model.eval()
    with torch.no_grad():
        pred = model(val_dataloader.x, val_dataloader.edge_index)
        if is_multilabel:
            preds = torch.sigmoid(pred)
            preds = (preds > 0.5).float()
            labels = val_dataloader.y.float()
            from sklearn.metrics import f1_score
            MCAccuracy = f1_score(labels.cpu(), preds.cpu(), average='micro')
            MCAccuracy = torch.tensor(MCAccuracy)
        else:
            metric = MulticlassAccuracy()
            metric.update(pred.argmax(-1), val_dataloader.y)
            MCAccuracy = metric.compute()

    return (model, MCAccuracy)


def test_on_testset(
    test_dataloader, model, device, is_multilabel=False,
):
    MCAccuracies = []

    for _ in range(3):
        transform = RandomNodeSplit(split="train_rest", num_val=0.33, num_test=0.33)
        test_data_split = transform(test_dataloader).to(device)
        test_data_split.subgraph(test_data_split["train_mask"])

        pred = model(test_data_split.x, test_data_split.edge_index)
        if is_multilabel:
            preds = torch.sigmoid(pred)
            preds = (preds > 0.5).float()
            labels = test_data_split.y.float()
            from sklearn.metrics import f1_score
            MCAccuracy = f1_score(labels.cpu(), preds.cpu(), average='micro')
            MCAccuracy = torch.tensor(MCAccuracy)
        else:
            metric = MulticlassAccuracy()
            metric.update(pred.argmax(-1), test_data_split.y)
            MCAccuracy = metric.compute()

        MCAccuracies.append(MCAccuracy)

    MCAccuracies = torch.stack(MCAccuracies)
    return torch.mean(MCAccuracies), torch.std(MCAccuracies)


def train_and_test_model(
    train_dataloader, val_dataloader, test_dataloader, model,
    layers, epochs, learning_rate, weight_decay, dropout,
    device,
    is_multilabel
):

    if is_multilabel:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    model_trained, _ = train_and_validate(
        train_dataloader, val_dataloader, model, loss_fn, layers, epochs, learning_rate,
        weight_decay,
        dropout, 0,
        is_multilabel=is_multilabel,
    )

    mean_acc, std_acc = test_on_testset(test_dataloader, model_trained, device, is_multilabel=is_multilabel)
    return model_trained, mean_acc, std_acc