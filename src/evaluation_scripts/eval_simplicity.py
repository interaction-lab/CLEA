"""
Plot linear preference alignment curves (simplicity evaluation).

By default, reads results/linear_results.csv (generate with generate_linear_results.py
after training your own models). Pass --pregenerated to instead read
pregenerated_results/linear_results.csv, the committed snapshot that reproduces the
paper's reported numbers without training anything.

Usage:
    python evaluation_scripts/eval_simplicity.py
    python evaluation_scripts/eval_simplicity.py --pregenerated
"""

import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_DIM = 8  # Which embedding size to plot

palette = {
    'contrastive': '#ff91af',
    'contrastive+autoencoder': '#e05780',
    'contrastive+VAE': '#f7cad0',
    'VAE': '#b6e2d3',
    'random': '#8f7073',
    'pretrained': '#aaaaaa',
    'autoencoder': '#86a79c',
}

method2label = {
    'random': 'Random',
    'pretrained': 'Pretrained',
    'autoencoder': 'AE',
    'VAE': 'VAE',
    'contrastive': 'CLEA',
    'contrastive+autoencoder': 'CLEA+AE',
    'contrastive+VAE': 'CLEA+VAE',
}


def main(pregenerated=False):
    results_dir = 'pregenerated_results' if pregenerated else 'results'
    plots_dir = os.path.join(_ROOT, results_dir, 'plots')
    data = pd.read_csv(os.path.join(_ROOT, results_dir, 'linear_results.csv'))
    data = data.query(f'dim_embedding == {EMBEDDING_DIM}')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, modality in zip(axes, ['visual', 'auditory', 'kinetic']):
        results = {m: [] for m in palette}
        mod_data = data.query(f'modality == "{modality}"')

        for _, row in mod_data.iterrows():
            m_vals = np.fromstring(row['m'][1:-1], sep=' ')
            if row['method'] in results:
                results[row['method']].append(m_vals)

        for method, values in results.items():
            if not values:
                continue
            color = palette[method]
            m = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            n = len(values)
            ax.fill_between(range(101), m - std / np.sqrt(n), m + std / np.sqrt(n),
                            alpha=0.2, color=color)
            ax.plot(m, label=method2label[method], color=color)
            if 'contrastive' in method:
                ax.plot(range(101)[::5], m[::5], color='#ffcc00',
                        linestyle=' ', marker='o', markersize=2.5)

        ax.set_axisbelow(True)
        ax.grid(color='#DEDEDE', linestyle='dashed')
        ax.set_title(modality.capitalize())
        ax.set_xlabel('Number of Queries')
        ax.set_ylim(-0.1, 0.65)
        ax.set_xlim(0, 100)

    axes[0].set_ylabel('Alignment')
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    plt.legend(ncol=7, bbox_to_anchor=(0.7, -0.15))
    plt.subplots_adjust(bottom=0.2, wspace=0.1, right=0.95, left=0.1)
    sns.despine()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, 'simplicity.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved -> {save_path}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pregenerated', action='store_true',
                        help='Read from pregenerated_results/ instead of results/.')
    args = parser.parse_args()
    main(args.pregenerated)
