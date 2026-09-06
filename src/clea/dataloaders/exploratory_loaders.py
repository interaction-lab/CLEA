import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}


def preload_data(stimulus_directory, stimulus_mapping, exploratory_action_data):
    """Pre-load all stimuli referenced in exploratory_action_data into memory."""
    print('loading data...')

    stim_ids = set()
    ids = stimulus_mapping['id'].values

    for i, row in exploratory_action_data.iterrows():
        for id in row['explored'].split(','):
            if int(id) in ids:
                stim_ids.add(int(id))
        for id in row['ignored'].split(','):
            if int(id) in ids:
                stim_ids.add(int(id))

    database = {}
    for id in tqdm(stim_ids):
        fname = stimulus_mapping.query(f'id=={id}')['file'].values[0]
        path = stimulus_directory + fname[:-4] + '.jpg'
        with Image.open(path) as im:
            im = np.array(im) / 255.0
            im = np.moveaxis(im, -1, 0)
            database[id] = im

    return database


class RawChoiceDataset(Dataset):
    """
    Dataset that loads raw stimuli (images or motion trajectories) and returns
    (anchor, positive, negative) triplets for contrastive / AE / VAE training.
    """

    def __init__(self, df, transform=None, kind='visual', data_dir='./data/', weighted=False):
        self.data = df
        self.transform = transform
        self.kind = kind
        self.indexer_csv_location = data_dir + 'all_data.csv'
        self.reweight = weighted

        if kind == 'visual':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Video"')
            self.stimulus_directory = data_dir + 'visual/vis/'
            self.stimulus_array = preload_data(self.stimulus_directory, self.stimulus_mapping, self.data)

        elif kind == 'auditory':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Audio"')
            self.stimulus_directory = data_dir + 'auditory/aud/'
            self.stimulus_array = preload_data(self.stimulus_directory, self.stimulus_mapping, self.data)

        elif kind == 'kinetic':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Movement"')
            self.stimulus_array = np.load(data_dir + 'kinetic/behaviors.npy')

    def get_stimulus_fname(self, index):
        if self.kind == 'visual':
            name = self.stimulus_mapping.query(f'id=={index}').file.values[0]
            return self.stimulus_directory + name.replace('mp4', 'jpg')
        elif self.kind == 'auditory':
            name = self.stimulus_mapping.query(f'id=={index}').file.values[0]
            return self.stimulus_directory + name.replace('wav', 'jpg')
        elif self.kind == 'kinetic':
            return int(index)

    def get_input_dim(self):
        if self.kind in ['auditory', 'visual']:
            im = Image.open(self.get_stimulus_fname(0))
            return np.moveaxis(np.array(im), -1, 0).shape
        elif self.kind == 'kinetic':
            return self.stimulus_array[self.get_stimulus_fname(0), :].shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data.iloc[item]
        selected, unselected = d['explored'].split(','), d['ignored'].split(',')

        anchor_set = selected
        negative_set = unselected

        if len(selected) == 1:
            anchor_set = unselected
            negative_set = selected
        elif np.random.rand() < 0.5 and len(unselected) > 1:
            anchor_set = unselected
            negative_set = selected

        anchor_id, positive_id = random.sample(anchor_set, 2)
        negative_id = random.sample(negative_set, 1)[0]

        if self.kind in ['auditory', 'visual']:
            anchor = self.stimulus_array[int(anchor_id)]
            positive = self.stimulus_array[int(positive_id)]
            negative = self.stimulus_array[int(negative_id)]
        elif self.kind == 'kinetic':
            anchor = self.stimulus_array[self.get_stimulus_fname(int(anchor_id)), :] * 25
            positive = self.stimulus_array[self.get_stimulus_fname(int(positive_id)), :] * 25
            negative = self.stimulus_array[self.get_stimulus_fname(int(negative_id)), :] * 25

        if self.reweight:
            return (self.transform(anchor), self.transform(positive),
                    self.transform(negative), torch.tensor(d['weight']))

        return self.transform(anchor), self.transform(positive), self.transform(negative)
