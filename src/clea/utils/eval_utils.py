"""
Utilities for generating embeddings from trained models.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}


def generate_all_embeddings_independent(
    model,
    dataframe,
    embedding_size: int,
    signal: str,
    device: str,
    data_dir: str = '../data',
    pretrained_embeds_array=None,
    embed_storage_path=None,
):
    """
    Generate embeddings for every stimulus in *dataframe* using a signal-specific
    (independent) model and store them in a (N, 4, dim) array.

    Args:
        model: Trained representation model with an ``encode`` method.
        dataframe: DataFrame with columns [id, type, file] from all_data.csv.
        embedding_size: Dimensionality of the output embedding.
        signal: Which signal slot to fill ('idle', 'searching', etc.).
        device: PyTorch device string.
        data_dir: Root data directory.
        pretrained_embeds_array: Optional pre-computed feature array (N, feat_dim).
        embed_storage_path: If provided, load an existing embedding array to fill
                            rather than starting from zeros.

    Returns:
        numpy array of shape (max_id + 1, 4, embedding_size).
    """
    model.eval()
    model.to(device)

    signal_index = TASK_INDEX_MAPPING[signal]

    if embed_storage_path is not None and os.path.exists(embed_storage_path):
        embeds = np.load(embed_storage_path)
    else:
        embeds = np.zeros((dataframe['id'].max() + 1, 4, embedding_size))

    trajectories = None
    if pretrained_embeds_array is None and (dataframe['type'] == 'Movement').any():
        trajectories = np.load(f"{data_dir}/kinetic/behaviors.npy")

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        stimulus_id = row['id']

        if pretrained_embeds_array is not None:
            x = torch.Tensor(pretrained_embeds_array[stimulus_id]).unsqueeze(0).to(device)
            out = model.encode(x)

        elif row['type'] == 'Video':
            im = np.array(Image.open(
                f"{data_dir}/visual/vis/{row['file'].replace('mp4', 'jpg')}"
            )) / 255.0
            im = np.moveaxis(im, -1, 0)
            out = model.encode(torch.Tensor(im).unsqueeze(0).to(device))

        elif row['type'] == 'Audio':
            im = np.array(Image.open(
                f"{data_dir}/auditory/aud/{row['file'].replace('wav', 'jpg')}"
            )) / 255.0
            im = np.moveaxis(im, -1, 0)
            out = model.encode(torch.Tensor(im).unsqueeze(0).to(device))

        elif row['type'] == 'Movement':
            traj = trajectories[stimulus_id] * 25
            out = model.encode(torch.Tensor(traj).unsqueeze(0).to(device))

        else:
            raise ValueError(f"Unknown stimulus type: {row['type']}")

        embeds[stimulus_id, signal_index, :] = out.detach().cpu().numpy()

    return embeds
