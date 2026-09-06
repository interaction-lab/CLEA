"""
Generate and cache embeddings for all stimuli from trained models.

For each trained model checkpoint found in results/trained_models/, this script:
  1. Loads the model.
  2. Runs all stimuli through the encoder.
  3. Saves a (N, 4, dim) embedding array to results/embeds/.

The embedding filename follows the same convention as the model checkpoint,
replacing the .pth extension with .npy.

Usage:
    python evaluation_scripts/generate_embeddings.py

Edit DATA_DIR and MODEL_DIR at the top of this file if needed.
"""

import os
import torch
import numpy as np
import pandas as pd

from clea.utils.eval_utils import generate_all_embeddings_independent

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, 'data')
MODEL_DIR = os.path.join(_ROOT, 'results', 'trained_models')
EMBED_DIR = os.path.join(_ROOT, 'results', 'embeds')

MODALITY_TO_TYPE = {
    'visual': 'Video',
    'auditory': 'Audio',
    'kinetic': 'Movement',
}

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    os.makedirs(EMBED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    all_data_df = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))

    for model_fname in sorted(os.listdir(MODEL_DIR)):
        if not model_fname.endswith('.pth'):
            continue

        # Expected format: {modality}&independent&raw&{method}&{signal}&{dim}.pth
        parts = model_fname[:-4].split('&')
        if len(parts) != 6:
            print(f'Skipping {model_fname} (unexpected filename format)')
            continue

        modality, task_dep, pretraining, method, signal, dim = parts
        embed_fname = f'{modality}&{task_dep}&{pretraining}&{method}&all_signals&{dim}.npy'
        embed_path = os.path.join(EMBED_DIR, embed_fname)

        print(f'Processing: {model_fname}')

        # weights_only=False: checkpoints are full pickled nn.Module objects saved
        # by the training scripts in this repo, not just state_dicts.
        model = torch.load(os.path.join(MODEL_DIR, model_fname),
                           map_location=torch.device(DEVICE), weights_only=False)
        model.eval()
        model.device = DEVICE
        model.to(DEVICE)

        stimulus_type = MODALITY_TO_TYPE.get(modality)
        if stimulus_type is None:
            print(f'  Unknown modality "{modality}", skipping.')
            continue

        df = all_data_df.query(f'type == "{stimulus_type}"')

        # Load existing array if already partially computed
        existing_path = embed_path if os.path.exists(embed_path) else None

        embeds = generate_all_embeddings_independent(
            model=model,
            dataframe=df,
            embedding_size=int(dim),
            signal=signal,
            device=DEVICE,
            data_dir=DATA_DIR,
            embed_storage_path=existing_path,
        )

        np.save(embed_path, embeds)
        print(f'  Saved -> {embed_path}  shape={embeds.shape}')


if __name__ == '__main__':
    main()
