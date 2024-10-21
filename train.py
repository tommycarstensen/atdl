import torch
from torcheval.metrics import MulticlassAccuracy
import torch.optim.adam
import torch_geometric.data.data
from typing import Callable
from torch_geometric.transforms import RandomNodeSplit
import wandb
from torch import nn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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
    train_loader,
    val_loader,
    device,
    model,
    loss_fn,
    optimizer,
    epochs,
    patience=10,
    wandb_iteration=0,
    wandb_toggle=False,
    is_multilabel=False,
):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            if is_multilabel:
                loss = loss_fn(pred, batch.y.float())
            else:
                loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_pred = model(batch.x, batch.edge_index)
                if is_multilabel:
                    val_loss = loss_fn(val_pred, batch.y.float())
                else:
                    val_loss = loss_fn(val_pred, batch.y)
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss}")
            model.load_state_dict(best_model_state)
            break

        # Optional logging
        if wandb_toggle:
            wandb.log({
                f'plot iteration {wandb_iteration}': {
                    'epoch': epoch + 1,
                    'train_loss': loss.item(),
                    'val_loss': avg_val_loss
                }
            })


def train_and_validate(
    train_dataloader, val_dataloader, model_function, loss_fn, layers, epochs, learning_rate, weight_decay, dropout, wandb_iteration, is_multilabel=False, wandb_toggle=False,
    ):
    model = model_function(layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loop_with_early_stopping(
        train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs,
        wandb_iteration=wandb_iteration,
        wandb_toggle=wandb_toggle,
        is_multilabel=is_multilabel,
        patience=100,
        )

    model.eval()
    with torch.no_grad():
        pred = model(val_dataloader.x, val_dataloader.edge_index)
        if is_multilabel:
            preds = torch.sigmoid(pred)
            preds = (preds > 0.5).float()
            labels = val_dataloader.y.float()
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
            MCAccuracy = f1_score(labels.cpu(), preds.cpu(), average='micro')
            MCAccuracy = torch.tensor(MCAccuracy)
        else:
            metric = MulticlassAccuracy()
            metric.update(pred.argmax(-1), test_data_split.y)
            MCAccuracy = metric.compute()

        MCAccuracies.append(MCAccuracy)

    MCAccuracies = torch.stack(MCAccuracies)
    return torch.mean(MCAccuracies), torch.std(MCAccuracies)


def test_on_testset_without_randomnodesplit(
    test_loader, model, device, is_multilabel=False,
):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index)

            if is_multilabel:
                preds = torch.sigmoid(pred)
                preds = (preds > 0.5).float()
                labels = batch.y.float()
            else:
                preds = pred.argmax(dim=-1)
                labels = batch.y

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if is_multilabel:
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        mean_acc = torch.tensor(micro_f1)
        std_acc = torch.tensor(0.0)  # Standard deviation not computed here
    else:
        mean_acc = torch.tensor(accuracy_score(all_labels, all_preds))
        std_acc = torch.tensor(0.0)  # Standard deviation not computed here

    return mean_acc, std_acc


def train_and_test_model(
    train_loader, val_loader, test_loader, model_function,
    layers, epochs, learning_rate, weight_decay, dropout,
    device,
    is_multilabel=False,
    wandb_toggle=False,
):
    if is_multilabel:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    model = model_function(layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loop_with_early_stopping(
        train_loader, val_loader, device, model, loss_fn, optimizer, epochs,
        patience=100,
        is_multilabel=is_multilabel,
        wandb_toggle=wandb_toggle,
    )

    mean_acc, std_acc = test_on_testset_without_randomnodesplit(
        test_loader, model, device, is_multilabel=is_multilabel)

    return model, mean_acc, std_acc