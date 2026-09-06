"""
Train kinetic (motion trajectory) representation models.

Trains one model per exploratory signal (idle, searching, has_item, has_information)
for the specified embedding type and saves checkpoints to results/trained_models/.

Usage:
    python training_scripts/train_kinetic.py

Hyperparameters are read from train_config.py -- edit that file before running.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from clea.dataloaders.exploratory_loaders import RawChoiceDataset
from clea.models.kinetic import RawSequenceEncoder, Seq2Seq, Seq2SeqVAE
from clea.utils.train_utils import train_single_epoch


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, 'data') + os.sep
MODEL_SAVE_DIR = os.path.join(_ROOT, 'results', 'trained_models') + os.sep
SIGNALS = ['idle', 'searching', 'has_item', 'has_information']


def _select_device() -> str:
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        torch.zeros(1, device='cuda:0')
        return 'cuda:0'
    except RuntimeError:
        print('Warning: CUDA device not compatible with this PyTorch build, falling back to CPU.')
        return 'cpu'


def get_dataloader(batch_size: int, signal: str):
    df = pd.read_csv(DATA_DIR + 'exploratory_action_data/plays_and_options.csv')
    df = df.query(f'type == "kinetic" & signal == "{signal}"')
    dataset = RawChoiceDataset(df, kind='kinetic', transform=torch.Tensor, data_dir=DATA_DIR)
    return DataLoader(dataset, batch_size=batch_size), dataset.get_input_dim()


def get_model_and_loss(model_type: str, latent_dim: int, margin: float, beta: float, device: str):
    """Return (model, loss_fn) for the given embedding type.

    For contrastive/random, returns the encoder with a triplet loss (or None for
    random).  For AE/VAE variants, returns the full Seq2Seq/Seq2SeqVAE with its
    reconstruction loss.  Contrastive+AE/VAE models use the AE/VAE architecture
    so that both a reconstruction and a triplet term can be applied during
    training.  margin sets the triplet margin (contrastive terms); beta sets
    the KL regularization weight (VAE terms).
    """
    if model_type == 'contrastive':
        return RawSequenceEncoder(input_size=3, hidden_size=latent_dim, num_layers=2,
                                  device=device), \
               nn.TripletMarginLoss(margin=margin, reduction='none')
    elif model_type == 'random':
        return RawSequenceEncoder(input_size=3, hidden_size=latent_dim, num_layers=2,
                                  device=device), None
    elif model_type == 'autoencoder':
        return Seq2Seq(input_size=3, hidden_size=latent_dim, num_layers=2,
                       device=device), nn.MSELoss()
    elif model_type == 'contrastive+autoencoder':
        return Seq2Seq(input_size=3, hidden_size=latent_dim, num_layers=2,
                       device=device), nn.MSELoss()
    elif model_type == 'VAE':
        model = Seq2SeqVAE(input_size=3, hidden_size=latent_dim, num_layers=2, device=device, beta=beta)
        return model, model.vae_loss
    elif model_type == 'contrastive+VAE':
        model = Seq2SeqVAE(input_size=3, hidden_size=latent_dim, num_layers=2, device=device, beta=beta)
        return model, model.vae_loss
    else:
        raise ValueError(f'Unknown model_type: {model_type}')


def train(model_type: str, device: str = 'cpu', batch_size: int = 128,
          embedding_dim: int = 128, lr: float = 1e-3, num_epochs: int = 300,
          margin: float = 2.0, beta: float = 10.0, skip_existing: bool = False):
    """
    Train a kinetic representation model for all four exploratory signals.
        The kinetic encoder architecture is a 2-layer LSTM that operates on sequences
        of robot joint states.

    Args:
        model_type: One of 'contrastive', 'autoencoder', 'VAE',
                    'contrastive+autoencoder', 'contrastive+VAE', 'random'.
        device: PyTorch device string.
        batch_size: Training batch size.
        embedding_dim: Hidden size / latent dimensionality of the GRU.
        lr: Adam learning rate.
        num_epochs: Number of training epochs.
        margin: Triplet margin (alpha) used by all contrastive terms.
        beta: KL regularization weight used by all VAE terms.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for signal in SIGNALS:
        save_path = os.path.join(
            MODEL_SAVE_DIR,
            f'kinetic&independent&raw&{model_type}&{signal}&{embedding_dim}.pth',
        )
        if skip_existing and os.path.exists(save_path):
            print(f'Skipping kinetic | signal={signal} | type={model_type} | dim={embedding_dim} (already exists)')
            continue

        print(f'Training kinetic | signal={signal} | type={model_type} | dim={embedding_dim}')

        data, _ = get_dataloader(batch_size, signal)
        model, loss_fn = get_model_and_loss(model_type, embedding_dim, margin, beta, device)

        if loss_fn is not None:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.train()
            model.to(device)

            for epoch in tqdm(range(num_epochs)):
                _, avg_loss = train_single_epoch(
                    embedding_type=model_type, model=model, loss_fn=loss_fn,
                    data_loader=data, optimizer=optimizer, epoch=epoch, device=device,
                    margin=margin,
                )
                tqdm.write(f'  Epoch {epoch:3d} | loss={avg_loss:.4f}')

        torch.save(model, save_path)
        print(f'  Saved -> {save_path}')


if __name__ == '__main__':
    # Hyperparameters come from train_config.py -- edit the shared block and the
    # KINETIC override dict there instead of this file.
    import train_config as cfg

    device = cfg.DEVICE or _select_device()

    for embedding_dim in cfg.EMBEDDING_DIMS:
        for model_type in cfg.MODEL_TYPES:
            train(model_type, device, cfg.BATCH_SIZE, embedding_dim, cfg.LR, cfg.NUM_EPOCHS,
                  cfg.KINETIC['margin'], cfg.KINETIC['beta'],
                  skip_existing=cfg.SKIP_EXISTING)
