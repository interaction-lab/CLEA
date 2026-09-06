import torch
import numpy as np
from torch.utils.data import Dataset


TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}


class UserStudyQueryDataloader(Dataset):
    """
    Loads pairwise preference queries from a user study.
    Returns two embeddings and a binary label (0 or 1) indicating the preferred option.
    """

    def __init__(self, query_array, embeddings_array, signal, transform=None):
        self.transform = transform
        self.embeddings = embeddings_array
        self.data = query_array  # (N, 2) array; second index is the preferred item
        self.signal_index = TASK_INDEX_MAPPING[signal]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        i1, i2 = self.data[item]
        e1 = self.embeddings[i1, self.signal_index]
        e2 = self.embeddings[i2, self.signal_index]
        e1, e2 = self.transform(e1), self.transform(e2)

        if np.random.rand() < 0.5:
            return e1, e2, torch.tensor(1)
        else:
            return e2, e1, torch.tensor(0)
