"""
Train every modality (visual, auditory, kinetic) from a single entry point.

All hyperparameters come from train_config.py. This
script just loops over EMBEDDING_DIMS x MODEL_TYPES for each modality and calls
that modality's existing train() function.

Usage:
    python training_scripts/train_all.py
    python training_scripts/train_all.py --modalities visual auditory
"""

import argparse

import torch

import train_config as cfg
import train_visual
import train_auditory
import train_kinetic


# Which train_config.py override dict feeds each modality's train(), and
# which of that dict's keys the modality's train() actually accepts.
MODALITIES = {
    'visual': (train_visual.train, cfg.VISUAL, ('margin', 'beta', 'hidden_dim')),
    'auditory': (train_auditory.train, cfg.AUDITORY, ('margin', 'beta', 'hidden_dim')),
    'kinetic': (train_kinetic.train, cfg.KINETIC, ('margin', 'beta')),
}


def _select_device() -> str:
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        torch.zeros(1, device='cuda:0')
        return 'cuda:0'
    except RuntimeError:
        print('Warning: CUDA device not compatible with this PyTorch build, falling back to CPU.')
        return 'cpu'


def main(modalities):
    device = cfg.DEVICE or _select_device()

    for name in modalities:
        train_fn, overrides, accepted_keys = MODALITIES[name]
        modality_kwargs = {k: v for k, v in overrides.items() if k in accepted_keys}

        for embedding_dim in cfg.EMBEDDING_DIMS:
            for model_type in cfg.MODEL_TYPES:
                train_fn(
                    model_type, device, cfg.BATCH_SIZE, embedding_dim, cfg.LR, cfg.NUM_EPOCHS,
                    skip_existing=cfg.SKIP_EXISTING, **modality_kwargs,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--modalities', nargs='+', choices=list(MODALITIES), default=list(MODALITIES),
        help='Which modalities to train (default: all three).',
    )
    args = parser.parse_args()
    main(args.modalities)
