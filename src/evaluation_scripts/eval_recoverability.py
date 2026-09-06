"""
Plot neural-network preference accuracy (recoverability evaluation).

By default, reads results/nn_results.csv (generate with generate_nn_results.py after
training your own models). Pass --pregenerated to instead read
pregenerated_results/nn_results.csv, the committed snapshot that reproduces the
paper's reported numbers without training anything.

Usage:
    python evaluation_scripts/eval_recoverability.py
    python evaluation_scripts/eval_recoverability.py --pregenerated
"""

import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_SIZE = 128  # Which embedding size to plot

name2label = {
    'random': 'Random',
    'pretrained': 'Pretrained',
    'autoencoder': 'AE',
    'VAE': 'VAE',
    'contrastive': 'CLEA',
    'contrastive+autoencoder': 'CLEA+AE',
    'contrastive+VAE': 'CLEA+VAE',
}

hue_order = [name2label[h] for h in
             ['random', 'pretrained', 'autoencoder', 'VAE',
              'contrastive', 'contrastive+autoencoder', 'contrastive+VAE']]

palette = {
    'CLEA': '#ff91af',
    'CLEA+AE': '#e05780',
    'CLEA+VAE': '#f7cad0',
    'VAE': '#b6e2d3',
    'Random': '#8f7073',
    'Pretrained': '#aaaaaa',
    'AE': '#86a79c',
}


def main(pregenerated=False):
    results_dir = 'pregenerated_results' if pregenerated else 'results'
    plots_dir = os.path.join(_ROOT, results_dir, 'plots')
    df = pd.read_csv(os.path.join(_ROOT, results_dir, 'nn_results.csv'))
    df['method'] = df['method'].map(name2label).fillna(df['method'])

    fig = plt.figure(figsize=(6.5, 5.5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    ax = sns.barplot(
        data=df.query(f'embedding_size == {EMBEDDING_SIZE}'),
        x='modality', y='accuracy',
        hue='method', hue_order=hue_order, palette=palette,
        capsize=0.1, errwidth=0.8, errorbar='se',
        order=['visual', 'auditory', 'kinetic'],
    )

    ax.set_axisbelow(True)
    ax.grid(color='#DEDEDE', linestyle='dashed')
    plt.xlabel('')
    plt.xticks([0, 1, 2], ['Visual', 'Auditory', 'Kinetic'])
    plt.ylabel('Test Preference Accuracy')
    plt.ylim(0.5, 1.0)

    reorder = lambda hl, nc: (sum((lis[i::nc] for i in range(nc)), []) for lis in hl)
    h_l = ax.get_legend_handles_labels()
    ax.legend(*reorder(h_l, 4), ncol=4, bbox_to_anchor=(0.5, 1.2), loc='upper center')

    sns.despine()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, 'recoverability.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved -> {save_path}')
    plt.show()

    print(df.query(f'embedding_size == {EMBEDDING_SIZE}')
            .groupby(['modality', 'method'])['accuracy'].mean().round(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pregenerated', action='store_true',
                        help='Read from pregenerated_results/ instead of results/.')
    args = parser.parse_args()
    main(args.pregenerated)
