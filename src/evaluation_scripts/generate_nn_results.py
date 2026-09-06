"""
Generate neural-network preference accuracy results (recoverability evaluation).

For each embedding file in results/embeds/ and each participant in
data/ranking_data/nn_reward_dataset/, this script:
  1. Trains a small RewardLearner MLP on the participant's training queries.
  2. Evaluates preference accuracy by predicting the choice between options.
  3. Appends the result to nn_results.csv.

Usage:
    python evaluation_scripts/generate_nn_results.py

Output: results/nn_results.csv
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from sklearn.random_projection import GaussianRandomProjection

from clea.dataloaders.query_loaders import UserStudyQueryDataloader
from clea.models.reward import RewardLearner

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(_ROOT, 'results', 'embeds')
PARTICIPANT_DIR = os.path.join(_ROOT, 'data', 'ranking_data', 'nn_reward_dataset')
DEVICE = 'cpu'

PRETRAINED_EMBEDS = {
    'visual': os.path.join(_ROOT, 'data', 'visual', 'xclip_embeds.npy'),
    'auditory': os.path.join(_ROOT, 'data', 'auditory', 'ast_embeds.npy'),
    'kinetic': os.path.join(_ROOT, 'data', 'kinetic', 'xclip_embeds.npy'),
}
PRETRAINED_DIM = 128
RANDOM_STATE = 42

TRAIN_EPOCHS = 60
HIDDEN_DIM = 256
LR = 1e-3
BATCH_SIZE = 16
L1_REG = 0.01


def train_and_eval(embeds, data, signal, em_size):
    reward_model = RewardLearner(em_size, hidden_dim=HIDDEN_DIM)
    reward_model.to(DEVICE)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        UserStudyQueryDataloader(data['train'], embeds, signal, transform=torch.Tensor),
        batch_size=BATCH_SIZE,
    )

    for _ in range(TRAIN_EPOCHS):
        reward_model.train()
        for option1, option2, choice in train_loader:
            optimizer.zero_grad()
            r1 = reward_model(option1.to(DEVICE))
            r2 = reward_model(option2.to(DEVICE))
            rewards = torch.cat((r1, r2), dim=1)
            loss = (loss_fn(rewards, choice.to(DEVICE))
                    + L1_REG * torch.norm(r1, p=1)
                    + L1_REG * torch.norm(r2, p=1))
            loss.backward()
            optimizer.step()

    reward_model.eval()
    eval_fn = Accuracy(task='multiclass', num_classes=2).to(DEVICE)
    eval_values = []

    eval_loader = DataLoader(
            UserStudyQueryDataloader(data['train'], embeds, signal, transform=torch.Tensor),
            batch_size=BATCH_SIZE,
        )
    
    for option1, option2, choice in eval_loader:
        r1 = reward_model(option1.to(DEVICE))
        r2 = reward_model(option2.to(DEVICE))
        rewards = torch.cat((r1, r2), dim=1)
        eval_values.append(eval_fn(rewards, choice.to(DEVICE)).item())

    return np.nanmean(eval_values)


def generate_pretrained_results():
    """Evaluate pretrained (xclip/AST) embeddings reduced via random projection 
    for fair comparison with learned embeddings of varying sizes."""
    results = []
    for embed_modality, embed_path in PRETRAINED_EMBEDS.items():
        if not os.path.exists(embed_path):
            print(f'Pretrained embed not found: {embed_path}, skipping.')
            continue

        raw_embeds = np.load(embed_path)
        rp = GaussianRandomProjection(n_components=PRETRAINED_DIM, random_state=RANDOM_STATE)
        reduced = rp.fit_transform(raw_embeds)
        # Tile to (N, 4, dim) to match the format expected by UserStudyQueryDataloader
        embeds = np.tile(reduced[:, np.newaxis, :], (1, 4, 1))

        print(f'Pretrained {embed_modality}: {raw_embeds.shape} -> {embeds.shape}')

        for participant_fname in sorted(os.listdir(PARTICIPANT_DIR)):
            if not participant_fname.endswith('.npz'):
                continue

            pid, modality, signal = participant_fname[:-4].split('&')

            if modality != embed_modality:
                continue

            data = np.load(os.path.join(PARTICIPANT_DIR, participant_fname))
            accuracy = train_and_eval(embeds, data, signal, PRETRAINED_DIM)

            results.append({
                'pid': pid,
                'modality': modality,
                'method': 'pretrained',
                'train_type': 'pretrained',
                'embedding_size': PRETRAINED_DIM,
                'accuracy': accuracy,
            })
            print(f'  pid={pid} modality={modality} method=pretrained '
                  f'dim={PRETRAINED_DIM} acc={accuracy:.3f}')

    return results


def main():
    results = []

    for embed_fname in sorted(os.listdir(EMBED_DIR)):
        if not embed_fname.endswith('.npy'):
            continue

        parts = embed_fname[:-4].split('&')
        if len(parts) != 6:
            continue

        embed_modality, _, train_type, method, _, em_size = parts
        em_size = int(em_size)

        embeds = np.load(os.path.join(EMBED_DIR, embed_fname))

        for participant_fname in sorted(os.listdir(PARTICIPANT_DIR)):
            if not participant_fname.endswith('.npz'):
                continue

            pid, modality, signal = participant_fname[:-4].split('&')

            if modality != embed_modality:
                continue

            data = np.load(os.path.join(PARTICIPANT_DIR, participant_fname))
            accuracy = train_and_eval(embeds, data, signal, em_size)

            results.append({
                'pid': pid,
                'modality': modality,
                'method': method,
                'train_type': train_type,
                'embedding_size': em_size,
                'accuracy': accuracy,
            })
            print(f'  pid={pid} modality={modality} method={method} '
                  f'dim={em_size} acc={accuracy:.3f}')

    results += generate_pretrained_results()

    out_path = os.path.join(_ROOT, 'results', 'nn_results.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f'\nSaved {len(results)} results to {out_path}')


if __name__ == '__main__':
    main()
