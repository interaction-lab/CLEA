"""
Epoch-level training loops for representation models.
"""

import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader


def train_single_epoch(
    embedding_type: str,
    model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    optimizer,
    epoch: int,
    device: str = 'cuda',
    margin: float = 1.0,
    weighted: bool = False,
):
    """
    Train a representation model for one epoch.

    Args:
        embedding_type: One of 'contrastive', 'autoencoder', 'VAE',
                        'contrastive+autoencoder', 'contrastive+VAE'.
        model: The representation model to train.
        loss_fn: Loss function appropriate for the embedding_type.
        data_loader: DataLoader yielding (anchor, positive, negative[, weight]) batches.
        optimizer: Optimizer for model parameters.
        epoch: Current epoch index (used only for progress logging).
        device: Device string ('cpu', 'cuda', 'cuda:0', …).
        margin: Triplet margin (only used for contrastive terms).
        weighted: If True, expects a 4th element (per-sample weight) from the loader.

    Returns:
        (total_samples_seen, average_loss_per_sample)
    """
    train_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        anchor, positive, negative = batch[0], batch[1], batch[2]
        weights = batch[3] if weighted else None

        optimizer.zero_grad()

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        a_embed = model(anchor)
        p_embed = model(positive)
        n_embed = model(negative)

        if embedding_type == 'contrastive':
            loss = loss_fn(a_embed, p_embed, n_embed)
            loss = loss.mean()

        elif embedding_type == 'autoencoder':
            loss = (loss_fn(a_embed, anchor)
                    + loss_fn(p_embed, positive)
                    + loss_fn(n_embed, negative))
            loss += (torch.linalg.vector_norm(a_embed, dim=1).mean()
                     + torch.linalg.vector_norm(p_embed, dim=1).mean()
                     + torch.linalg.vector_norm(n_embed, dim=1).mean())

        elif embedding_type == 'VAE':
            loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative)

        elif embedding_type == 'contrastive+autoencoder':
            loss = (loss_fn(a_embed, anchor)
                    + loss_fn(p_embed, positive)
                    + loss_fn(n_embed, negative))
            trip_loss = nn.TripletMarginLoss(margin=margin, reduction='none')(
                model.encode(anchor), model.encode(positive), model.encode(negative)
            )
            if weighted and weights is not None:
                trip_loss = trip_loss * weights.to(device)
            loss += trip_loss.mean()

        elif embedding_type == 'contrastive+VAE':
            loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative)
            trip_loss = nn.TripletMarginLoss(margin=margin, reduction='none')(
                model.encode(anchor), model.encode(positive), model.encode(negative)
            )
            if weighted and weights is not None:
                trip_loss = trip_loss * weights.to(device)
            loss += trip_loss.mean()

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return epoch * len(data_loader.dataset), train_loss / len(data_loader.dataset)
