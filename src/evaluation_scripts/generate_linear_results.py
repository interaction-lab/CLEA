"""
Generate linear preference alignment results (simplicity / minimality evaluation).

Uses the IRL preference-learning library to simulate a Bayesian linear reward
learner receiving sequential queries and measures alignment with the participant's
true preference vector at each step.

Requires the irlpreference package:
    cd clea/preference-learning-from-selection && pip install -e .

Usage:
    python evaluation_scripts/generate_linear_results.py

Output: results/linear_results.csv
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.random_projection import GaussianRandomProjection

from irlpreference.input_models import LuceShepardChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(_ROOT, 'results', 'embeds')
PARTICIPANT_DIR = os.path.join(_ROOT, 'data', 'ranking_data', 'linear_reward_dataset')

PRETRAINED_EMBEDS = {
    'visual': os.path.join(_ROOT, 'data', 'visual', 'xclip_embeds.npy'),
    'auditory': os.path.join(_ROOT, 'data', 'auditory', 'ast_embeds.npy'),
    'kinetic': os.path.join(_ROOT, 'data', 'kinetic', 'xclip_embeds.npy'),
}
RANDOM_STATE = 42

TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}

NUM_TRIALS = 100      # Monte Carlo trials per participant
MAX_QUERIES = 100     # Maximum queries per simulation
NUM_SAMPLES = 10_000  # Reward posterior samples


def alignment_metric(true_w, guessed_w):
    """Cosine similarity between the true and estimated reward vectors."""
    denom = np.linalg.norm(guessed_w) * np.linalg.norm(true_w)
    if denom == 0:
        return 0.0
    return np.dot(guessed_w, true_w) / denom


def run_participant(embeds, data, signal, dim_embedding):
    """Simulate preference learning for one participant and return an alignment curve."""
    signal_index = TASK_INDEX_MAPPING[signal]
    true_preference = embeds[data['top_id'], signal_index]
    train_data = data['train']

    user_choice_model = LuceShepardChoice()
    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=NUM_SAMPLES)

    cumulative_values = []
    for _ in tqdm(range(NUM_TRIALS), leave=False):
        user_estimate.reset()
        alignment = [0]
        shuffled = np.random.permutation(train_data)

        for trial in shuffled[:MAX_QUERIES]:
            query = embeds[trial, signal_index]
            user_choice_model.tell_input(1, query)
            user_estimate.update(user_choice_model.get_probability_of_input)
            alignment.append(alignment_metric(user_estimate.get_expectation(), true_preference))

        cumulative_values.append(alignment)

    return np.mean(cumulative_values, axis=0), np.std(cumulative_values, axis=0)


def generate_pretrained_results():
    """Evaluate pretrained (xclip/AST) embeddings reduced via random projection."""
    results = []

    for dim_embedding in [8, 16, 32, 64, 128]:
        for embed_modality, embed_path in PRETRAINED_EMBEDS.items():
            if not os.path.exists(embed_path):
                print(f'Pretrained embed not found: {embed_path}, skipping.')
                continue

            raw_embeds = np.load(embed_path)
            rp = GaussianRandomProjection(n_components=dim_embedding, random_state=RANDOM_STATE)
            reduced = rp.fit_transform(raw_embeds)
            # Tile to (N, 4, dim) to match the format expected by run_participant
            embeds = np.tile(reduced[:, np.newaxis, :], (1, 4, 1))

            for participant_fname in sorted(os.listdir(PARTICIPANT_DIR)):
                if not participant_fname.endswith('.npy') and not participant_fname.endswith('.npz'):
                    continue

                name_parts = participant_fname[:-4].split('&')
                if len(name_parts) != 3:
                    continue

                pid, modality, signal = name_parts
                if modality != embed_modality:
                    continue

                data = np.load(os.path.join(PARTICIPANT_DIR, participant_fname))
                m, std = run_participant(embeds, data, signal, dim_embedding)

                results.append({
                    'method': 'pretrained',
                    'train_type': 'pretrained',
                    'dim_embedding': dim_embedding,
                    'm': m,
                    'std': std,
                    'modality': modality,
                    'signal': signal,
                    'pid': pid,
                })
                print(f'  method=pretrained modality={modality} dim={dim_embedding} '
                      f'pid={pid} final_alignment={m[-1]:.3f}')

    return results


def main():
    results = []

    for dim_embedding in [8, 16, 32, 64, 128]:
        for embed_fname in sorted(os.listdir(EMBED_DIR)):
            if not embed_fname.endswith('.npy'):
                continue

            parts = embed_fname[:-4].split('&')
            if len(parts) != 6:
                continue

            embed_modality, _, train_type, method, _, em_size = parts
            if int(em_size) != dim_embedding:
                continue

            embeds = np.load(os.path.join(EMBED_DIR, embed_fname))

            for participant_fname in sorted(os.listdir(PARTICIPANT_DIR)):
                if not participant_fname.endswith('.npy') and not participant_fname.endswith('.npz'):
                    continue

                name_parts = participant_fname[:-4].split('&')
                if len(name_parts) != 3:
                    continue

                pid, modality, signal = name_parts

                if modality not in ('visual', 'auditory', 'kinetic'):
                    continue

                if modality != embed_modality:
                    continue

                data = np.load(os.path.join(PARTICIPANT_DIR, participant_fname))
                m, std = run_participant(embeds, data, signal, dim_embedding)

                results.append({
                    'method': method,
                    'train_type': train_type,
                    'dim_embedding': dim_embedding,
                    'm': m,
                    'std': std,
                    'modality': modality,
                    'signal': signal,
                    'pid': pid,
                })
                print(f'  method={method} modality={modality} dim={dim_embedding} '
                      f'pid={pid} final_alignment={m[-1]:.3f}')

    results += generate_pretrained_results()

    out_path = os.path.join(_ROOT, 'results', 'linear_results.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f'\nSaved {len(results)} results to {out_path}')


if __name__ == '__main__':
    main()
