"""
Statistical significance testing for completeness (Fig. 5) and simplicity (Table I).

For each modality (and, for simplicity, each embedding dimension), runs
Bonferroni-corrected pairwise repeated-measures t-tests (one row per participant)
between the top-performing method and every other method. A method is reported
as significantly best only if it beats *all* other methods at p < .05, matching
the "(all p < .05)" / asterisk annotations used in the paper.

By default, reads results/nn_results.csv and results/linear_results.csv (generate
with generate_nn_results.py / generate_linear_results.py after training your own
models). Pass --pregenerated to instead read from pregenerated_results/, the
committed snapshot that reproduces the paper's reported numbers without training
anything.

Usage:
    python evaluation_scripts/eval_stats.py
    python evaluation_scripts/eval_stats.py --pregenerated
"""

import argparse
import os
import numpy as np
import pandas as pd
import pingouin as pg

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ['random', 'pretrained', 'autoencoder', 'VAE',
           'contrastive', 'contrastive+autoencoder', 'contrastive+VAE']


def get_auc(m_str):
    m = np.fromstring(m_str[1:-1], sep=' ')
    i_max = np.argmax(m)
    return np.trapz(m[:i_max + 1], dx=1 / 100)


def best_method(df, dv, subject='pid'):
    """Return the method with the highest mean `dv`, and whether it beats every
    other method at Bonferroni-corrected p < .05 (paired t-test, common subjects only)."""
    means = df.groupby('method')[dv].mean()
    means = means.reindex([m for m in METHODS if m in means.index])
    top_method = means.idxmax()

    pairwise = pg.pairwise_tests(data=df, dv=dv, within='method', subject=subject,
                                  padjust='bonf')
    # pingouin column naming differs across versions ('p-corr'/'p_corr'); 'p-corr'/'p_corr'
    # is only emitted when more than 2 groups are compared, so fall back to the
    # uncorrected p-value (a single comparison needs no multiple-comparison correction).
    p_col = next((c for c in ('p-corr', 'p_corr', 'p-unc', 'p_unc') if c in pairwise.columns))

    significant = True
    for _, row in pairwise.iterrows():
        if top_method not in (row['A'], row['B']):
            continue
        other = row['B'] if row['A'] == top_method else row['A']
        if means.get(other, -np.inf) >= means[top_method]:
            continue
        if row[p_col] >= 0.05:
            significant = False
            break

    return top_method, significant, means.round(3)


def completeness_significance(pregenerated=False):
    print('=== Completeness (Test Preference Accuracy, embedding_size=128) ===')
    results_dir = 'pregenerated_results' if pregenerated else 'results'
    df = pd.read_csv(os.path.join(_ROOT, results_dir, 'nn_results.csv'))
    df = df.query('embedding_size == 128')

    for modality in ['visual', 'auditory', 'kinetic']:
        mod_df = df.query('modality == @modality')
        # Only keep participants with a value for every method (required for paired tests)
        complete_pids = mod_df.groupby('pid')['method'].nunique()
        complete_pids = complete_pids[complete_pids == mod_df['method'].nunique()].index
        mod_df = mod_df[mod_df['pid'].isin(complete_pids)]

        if mod_df.empty:
            print(f'{modality}: no complete data, skipping.')
            continue

        top, significant, means = best_method(mod_df, dv='accuracy')
        marker = '*' if significant else ''
        print(f'\n{modality}:\n{means}')
        print(f'  Best: {top}{marker}')


def simplicity_significance(pregenerated=False):
    print('\n=== Simplicity (AUC-Alignment) ===')
    results_dir = 'pregenerated_results' if pregenerated else 'results'
    df = pd.read_csv(os.path.join(_ROOT, results_dir, 'linear_results.csv'))
    df['auc'] = df['m'].apply(get_auc)

    for modality in ['visual', 'auditory', 'kinetic']:
        for dim in [8, 16, 32, 64, 128]:
            mod_df = df.query('modality == @modality and dim_embedding == @dim')
            complete_pids = mod_df.groupby('pid')['method'].nunique()
            complete_pids = complete_pids[complete_pids == mod_df['method'].nunique()].index
            mod_df = mod_df[mod_df['pid'].isin(complete_pids)]

            if mod_df.empty:
                continue

            top, significant, means = best_method(mod_df, dv='auc')
            marker = '*' if significant else ''
            print(f'{modality} dim={dim:3d}: best={top}{marker}\t' +
                  '\t'.join(f'{m}={v}' for m, v in means.items()))


def main(pregenerated=False):
    completeness_significance(pregenerated)
    simplicity_significance(pregenerated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pregenerated', action='store_true',
                        help='Read from pregenerated_results/ instead of results/.')
    args = parser.parse_args()
    main(args.pregenerated)
