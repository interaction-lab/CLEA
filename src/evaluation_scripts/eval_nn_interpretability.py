"""
Plot nearest-exemplar cosine similarity (interpretability evaluation).

For each method, computes the cosine similarity between a participant's chosen
action embedding and the closest expert-designed exemplar embedding, then
subtracts a random-other-stimulus baseline (mean cosine similarity to N_RANDOM
randomly sampled stimuli, same signal) before plotting a grouped bar chart
across modalities.

The raw (un-normalized) exemplar similarity rewards representation collapse: if
a method's embedding space maps most stimuli to nearly the same point, the
chosen action and every exemplar all end up "close" regardless of any genuine
interpretable structure. This is most visible for the untrained 'random'
baseline, which topped the raw metric on visual/auditory (where its untrained
CNN output collapses hard) despite carrying no real signal. Subtracting the
random-other-stimulus baseline isolates the exemplar-specific component: how
much closer is the chosen action to a *relevant* exemplar than to an arbitrary
other stimulus in the same space. 'random' correctly falls to ~0 under this
metric.

Requires exemplar data at data/evaluation/concatenated_final_signals.csv and study-2
ranking results at data_collection/results/*.csv. Per-signal embeddings are read from
results/embeds by default (generate with generate_embeddings.py after training your
own models); pass --pregenerated to instead read pregenerated_results/embeds, the
committed snapshot (ships only one example file, not the full set this script needs,
so it will likely raise FileNotFoundError until you generate the rest).

Usage:
    python evaluation_scripts/eval_nn_interpretability.py
    python evaluation_scripts/eval_nn_interpretability.py --pregenerated
"""

import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, 'data', 'ranking_data', 'results')
EXEMPLAR_CSV = os.path.join(_ROOT, 'data', 'evaluation', 'concatenated_final_signals.csv')
PRETRAINED_EMBEDS = {
    'kinetic': os.path.join(_ROOT, 'data', 'kinetic', 'xclip_embeds.npy'),
    'visual': os.path.join(_ROOT, 'data', 'visual', 'xclip_embeds.npy'),
    'auditory': os.path.join(_ROOT, 'data', 'auditory', 'ast_embeds.npy'),
}

TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}
EMBEDDING_SIZE = 128  # Which embedding size to plot
N_RANDOM = 30  # Random other-stimulus draws used for the collapse-correction baseline
RNG = np.random.default_rng(0)

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


def load_study2_results():
    results = []
    for f in os.listdir(RESULTS_DIR):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(RESULTS_DIR, f))
        for _, row in df.query('rank == 4 and trial == 9').iterrows():
            results.append({'id': row['id'], 'modality': row['modality'], 'signal': row['signal']})
    return results


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom


def random_other_stimulus_baseline(embeds, sig_idx, query_id, query_vec):
    """Mean cosine similarity between query_vec and N_RANDOM random other
    stimuli (same signal), used to correct the exemplar-similarity metric for
    representation collapse -- see module docstring.
    """
    n_stimuli = embeds.shape[0]
    other_ids = RNG.integers(0, n_stimuli, N_RANDOM)
    other_ids = other_ids[other_ids != query_id]
    other_vecs = embeds[other_ids, sig_idx]
    sims = np.array([cosine_similarity(ov, query_vec) for ov in other_vecs])
    return sims.mean()


def main(pregenerated=False):
    results_dir = 'pregenerated_results' if pregenerated else 'results'
    embed_dir = os.path.join(_ROOT, results_dir, 'embeds')
    plots_dir = os.path.join(_ROOT, results_dir, 'plots')

    study2_results = load_study2_results()
    exemplar_data = pd.read_csv(EXEMPLAR_CSV)
    min_distances = []

    methods = ['random', 'contrastive+autoencoder', 'VAE', 'autoencoder',
               'contrastive', 'contrastive+VAE']

    for method in methods:
        for modality in ['visual', 'auditory', 'kinetic']:
            for result in study2_results:
                if result['modality'] != modality:
                    continue

                exemplars = exemplar_data.query('signal == @result["signal"]')[modality].values
                exemplars = exemplars[exemplars >= 0].astype(int)

                embeds = np.load(
                    os.path.join(embed_dir,
                                 f'{modality}&independent&raw&{method}&all_signals&{EMBEDDING_SIZE}.npy')
                )
                sig_idx = TASK_INDEX_MAPPING[result['signal']]
                query_vec = embeds[result['id'], sig_idx]
                exemplar_vecs = embeds[exemplars, sig_idx]

                sims = np.array([cosine_similarity(ev, query_vec) for ev in exemplar_vecs])
                best = np.max(sims)
                baseline = random_other_stimulus_baseline(embeds, sig_idx, result['id'], query_vec)

                min_distances.append({'dist': best - baseline, 'method': method, 'modality': modality})

    # Pretrained baselines
    for modality in ['visual', 'auditory', 'kinetic']:
        for result in study2_results:
            if result['modality'] != modality:
                continue

            exemplars = exemplar_data.query('signal == @result["signal"]')[modality].values
            exemplars = exemplars[exemplars >= 0].astype(int)

            raw_embeds = np.load(PRETRAINED_EMBEDS[modality])
            embeds = np.tile(raw_embeds[:, np.newaxis, :], (1, 4, 1))

            sig_idx = TASK_INDEX_MAPPING[result['signal']]
            query_vec = embeds[result['id'], sig_idx]
            exemplar_vecs = embeds[exemplars, sig_idx]

            sims = np.array([cosine_similarity(ev, query_vec) for ev in exemplar_vecs])
            best = np.max(sims)
            baseline = random_other_stimulus_baseline(embeds, sig_idx, result['id'], query_vec)

            min_distances.append({'dist': best - baseline, 'method': 'pretrained', 'modality': modality})

    df = pd.DataFrame(min_distances)
    df['method'] = df['method'].map(name2label).fillna(df['method'])

    fig = plt.figure(figsize=(6.5, 5.5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    ax = sns.barplot(
        data=df, x='modality', y='dist',
        hue='method', hue_order=hue_order, palette=palette,
        capsize=0.1, errwidth=0.8, errorbar='se',
        order=['visual', 'auditory', 'kinetic'],
    )

    ax.set_axisbelow(True)
    ax.grid(color='#DEDEDE', linestyle='dashed')
    plt.xlabel('')
    plt.ylabel('Normalized Score')
    # Scale to what the bars + SE error bars actually show, not the raw
    # per-participant range (which includes outliers the plot never displays).
    grouped = df.groupby(['modality', 'method'])['dist'].agg(['mean', 'sem'])
    y_min = min(0.0, (grouped['mean'] - grouped['sem']).min())
    y_max = (grouped['mean'] + grouped['sem']).max()
    plt.ylim(y_min, y_max + (y_max - y_min) * 0.1)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticklabels(['Visual', 'Auditory', 'Kinetic'])

    reorder = lambda hl, nc: (sum((lis[i::nc] for i in range(nc)), []) for lis in hl)
    h_l = ax.get_legend_handles_labels()
    ax.legend(*reorder(h_l, 4), ncol=4, bbox_to_anchor=(0.5, 1.2), loc='upper center')

    sns.despine()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, 'interpretability.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved -> {save_path}')
    plt.show()

    print(df.groupby('method')['dist'].mean().round(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pregenerated', action='store_true',
                        help='Read from pregenerated_results/ instead of results/.')
    args = parser.parse_args()
    main(args.pregenerated)
