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
    patience=100,
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
            out = model(batch.x, batch.edge_index)

            # Determine training indices
            if hasattr(batch, 'train_mask'):
                train_indices = batch.train_mask
            else:
                # Use all nodes in the batch
                train_indices = torch.arange(batch.num_nodes, device=device)

            # Compute loss on the appropriate nodes
            if is_multilabel:
                loss = loss_fn(out[train_indices], batch.y[train_indices].float())
            else:
                loss = loss_fn(out[train_indices], batch.y[train_indices])

            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)

                # Determine validation indices
                if hasattr(batch, 'val_mask'):
                    val_indices = batch.val_mask
                else:
                    # Use all nodes in the batch
                    val_indices = torch.arange(batch.num_nodes, device=device)

                # Compute validation loss
                if is_multilabel:
                    val_loss += loss_fn(out[val_indices], batch.y[val_indices].float()).item()
                else:
                    val_loss += loss_fn(out[val_indices], batch.y[val_indices]).item()

        avg_val_loss = val_loss / len(val_loader)

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
    # train_dataloader, val_dataloader,
    loader,
    model_function, loss_fn, layers, epochs, learning_rate, weight_decay, dropout, wandb_iteration, is_multilabel=False, wandb_toggle=False,
    ):
    model = model_function(layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loop_with_early_stopping(
        # train_dataloader, val_dataloader,
        loader,
        model, loss_fn, optimizer, epochs,
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
    test_loader,
    model, device, is_multilabel=False,
    # "Accuracy and standard deviation are computed from 3 random data splits."
    num_splits=3,
):

    model.eval()
    total_accuracy_or_micro_f1 = 0
    num_batches = 0
    accuracies = []

    with torch.no_grad():
        # Perform evaluation over the dataset for each split
        for split_num in range(num_splits):
            # Reinitialize random split here for each split_num
            # Random split for multi-graph datasets
            torch.manual_seed(42 + split_num)  # Ensure different random split each time
            for batch in test_loader:
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
                    # Apply sigmoid activation for multi-label classification
                    preds = torch.sigmoid(out[test_indices])
                    # Binarize predictions
                    preds = (preds > 0.5).float()
                    labels = batch.y[test_indices].float()
                    # Compute micro-F1 score for the current batch
                    micro_f1 = f1_score(labels.cpu(), preds.cpu(), average='micro')
                    total_accuracy_or_micro_f1 += micro_f1
                    num_batches += 1
                else:
                    preds = out[test_indices].argmax(dim=1)
                    labels = batch.y[test_indices]
                    # Compute accuracy for the current batch
                    acc = accuracy_score(labels.cpu(), preds.cpu())
                    total_accuracy_or_micro_f1 += acc
                    num_batches += 1

            # Store accuracy for each split
            accuracies.append(total_accuracy_or_micro_f1 / num_batches)

    # Convert accuracies to tensor to compute mean and std deviation
    accuracies_tensor = torch.tensor(accuracies)

    mean_acc = accuracies_tensor.mean().item()
    std_acc = accuracies_tensor.std().item()

    return mean_acc, std_acc



def train_and_test_model(
    train_loader, val_loader, test_loader,
    model_function,
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
        train_loader, val_loader,
        device, model, loss_fn, optimizer, epochs,
        patience=100,
        is_multilabel=is_multilabel,
        wandb_toggle=wandb_toggle,
    )

    mean_acc, std_acc = test_on_testset(
        test_loader,
        model, device, is_multilabel=is_multilabel)

    return model, mean_acc, std_acc