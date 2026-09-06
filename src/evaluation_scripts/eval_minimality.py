"""
Plot alignment AUC vs. embedding dimensionality (minimality evaluation).

By default, reads results/linear_results.csv (generate with generate_linear_results.py
after training your own models). Pass --pregenerated to instead read
pregenerated_results/linear_results.csv, the committed snapshot that reproduces the
paper's reported numbers without training anything.

Usage:
    python evaluation_scripts/eval_minimality.py
    python evaluation_scripts/eval_minimality.py --pregenerated
"""

import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

palette = {
    'contrastive': 'tab:orange',
    'contrastive+autoencoder': '#d62b0d',
    'contrastive+VAE': '#decb3a',
    'VAE': '#3399ff',
    'random': '#999999',
    'pretrained': '#aaaaaa',
    'autoencoder': '#0033cc',
}


def get_auc(m_str):
    m = np.fromstring(m_str[1:-1], sep=' ')
    i_max = np.argmax(m)
    return np.trapz(m[:i_max + 1], dx=1 / 100)


def main(pregenerated=False):
    results_dir = 'pregenerated_results' if pregenerated else 'results'
    plots_dir = os.path.join(_ROOT, results_dir, 'plots')
    df = pd.read_csv(os.path.join(_ROOT, results_dir, 'linear_results.csv'))

    plot_df = []
    for _, row in df.iterrows():
        m = np.fromstring(row['m'][1:-1], sep=' ')
        std = np.fromstring(row['std'][1:-1], sep=' ')
        i_max = np.argmax(m)
        auc = np.trapz(m[:i_max + 1], dx=1 / 100)
        plot_df.append({
            'method': row['method'],
            'embedding_size': int(row['dim_embedding']),
            'alignment': m[-1],
            'std': std[-1],
            'auc': auc,
            'modality': row['modality'],
        })

    plot_df = pd.DataFrame(plot_df)

    for modality in ['visual', 'auditory', 'kinetic']:
        plt.figure()
        mod_df = plot_df.query(f'modality == "{modality}"')
        ax = sns.lineplot(data=mod_df, x='embedding_size', y='auc', hue='method',
                          err_style='bars', palette=palette, errorbar='se')
        plt.title(modality.capitalize())
        plt.xlabel('Embedding Size')
        plt.ylabel('Alignment AUC')
        plt.legend(bbox_to_anchor=(1.05, 1.15), ncol=3)
        plt.tight_layout()

        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f'minimality_{modality}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved -> {save_path}')
        plt.show()

        print(f'\n{modality}')
        for method in ['random', 'pretrained', 'autoencoder', 'VAE',
                       'contrastive', 'contrastive+autoencoder', 'contrastive+VAE']:
            vals = (mod_df.query(f'method == "{method}"')
                          .groupby('embedding_size')
                          .mean(numeric_only=True)['auc']
                          .round(3).values)
            vals_str = [str(v).replace('-0.', '-.').replace('0.', '.') for v in vals]
            print(f'{method}:\t' + '\t'.join(vals_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pregenerated', action='store_true',
                        help='Read from pregenerated_results/ instead of results/.')
    args = parser.parse_args()
    main(args.pregenerated)
