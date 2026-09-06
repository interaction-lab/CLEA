"""
Central hyperparameter configuration for all training scripts.

Edit the values below to change hyperparameters for every modality. train_all.py
and each per-modality script (train_visual.py / train_auditory.py /
train_kinetic.py) read everything from here, so this one file configures both
the combined sweep and any individual script run on its own.

The defaults below reproduce the paper's reported numbers.
"""

# Shared across all three modalities.
DEVICE = None  # None = auto-select CUDA if available, otherwise CPU
BATCH_SIZE = 128
EMBEDDING_DIMS = [8, 16, 32, 64, 128]  # Table I / Fig. 6 sweep these; Fig. 5 and Fig. 7 use 128
LR = 1e-3
NUM_EPOCHS = 300
MODEL_TYPES = ['contrastive', 'autoencoder', 'VAE',
               'contrastive+autoencoder', 'contrastive+VAE', 'random']
SKIP_EXISTING = True  # set to False to retrain existing checkpoints

# Per-modality overrides. beta and margin differ by modality; only
# visual/auditory take a hidden_dim (kinetic's GRU has no separate hidden
# layer -- its embedding_dim doubles as the GRU hidden size).
VISUAL = dict(margin=0.1, beta=1.0, hidden_dim=512)
AUDITORY = dict(margin=0.1, beta=10.0, hidden_dim=512)
KINETIC = dict(margin=2.0, beta=10.0)
