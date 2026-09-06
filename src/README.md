# CLEA: Contrastive Learning from Exploratory Actions

Code for learning robot behavior representations from human exploratory action data. The robot in our study was designed to help users find items around the room, and users specified their preferences for signals that the robot made during this task. The robot had four state-expressive signals: idle, searching, has_item, and has_information. Each signal consisted of three modalities: a **visual** behavior (sequence of images), an **auditory** behavior (sequence of frequencies), and **kinetic** behavior (sequence of joint states). We performed experiments that compared four training objectives for representation learning: Contrastive, Autoencoder (AE), Variational Autoencoder (VAE), and hybrid combinations.

---

# Quick Start

### Install

```bash
conda create -n clea python=3.8
conda activate clea
pip install -e .
pip install -r requirements.txt

# Also required for the linear/Bayesian evaluation scripts:
cd clea/preference-learning-from-selection && pip install -e . && cd ../..
```

(`pip install -e .` / `pip install -r requirements.txt` without conda works too, if you'd rather manage the environment yourself.)

### Regenerate paper's figures from pre-computed representations

```bash
python evaluation_scripts/eval_recoverability.py --pregenerated       # Fig. 5
python evaluation_scripts/eval_simplicity.py --pregenerated           # Fig. 6
python evaluation_scripts/eval_minimality.py --pregenerated           # Table I
python evaluation_scripts/eval_stats.py --pregenerated                # significance tests
```

These read the committed `pregenerated_results/` snapshot, so they work without running anything else first.

### Or run the full pipeline yourself, in order

```bash
python training_scripts/train_all.py       # 1. Train representation models
python evaluation_scripts/evaluate_all.py  # 2. Generate embeddings + result tables
python evaluation_scripts/eval_recoverability.py  # 3. Plot (one script per figure/table)
```

Each stage outputs to `results/` and the subsequent stages read from results to perform their copmutation. See [Running Individual
Experiments](#running-individual-experiments) for additional details on what each stage does and how you can configure each stage.

---

# Technical Details

## Data Layout

```
data/
├── all_data.csv                             # Stimulus index (id, type, file)
├── Attributions.md                          # Sources/licensing for the raw signal media
├── exploratory_action_data/                 # Study 1: exploratory action choices
│   └── plays_and_options.csv
├── ranking_data/                            # Study 2: human preference/ranking data
│   ├── linear_reward_dataset/               # Per-participant queries for the linear/Bayesian reward eval (.npz)
│   ├── nn_reward_dataset/                   # Per-participant queries for the NN reward eval (.npz)
│   └── results/                             # Ranking study CSVs
├── visual/
│   ├── vis/                                 # JPEG frames (one per video clip)
│   └── xclip_embeds.npy                     # Pretrained X-CLIP features
├── auditory/
│   ├── aud/                                 # JPEG spectrograms (one per audio clip)
│   └── ast_embeds.npy                       # Pretrained AST features
├── kinetic/
│   ├── kin/                                 # Rendered clips of each motion trajectory (not used by training/eval code -- browsing only)
│   ├── behaviors.npy                        # (N, T, 3) array of motion trajectories
│   └── xclip_embeds.npy                     # Pretrained X-CLIP features
└── evaluation/
    └── concatenated_final_signals.csv       # Expert-designed exemplar mappings
```

Fresh training/evaluation output is written to `results/` (git-ignored, regenerated
locally by running the pipeline below):

```
results/
├── trained_models/    # Model checkpoints (.pth) saved by training scripts
├── embeds/            # Cached embedding arrays (.npy) saved by generate_embeddings.py
├── nn_results.csv     # Saved by generate_nn_results.py
├── linear_results.csv # Saved by generate_linear_results.py
└── plots/             # PNGs saved by the eval_*.py plotting scripts
```

`pregenerated_results/` holds the data from the paper -- the `nn_results.csv` and
`linear_results.csv` that reproduce the paper's reported numbers, plus a small
example set of checkpoints/embeds (not the full sweep). The plotting scripts read
from `results/` by default; pass `--pregenerated` to read from this snapshot instead
(and save plots to `pregenerated_results/plots/`), without training or generating
anything.

## Representation Learning Structure

The pipeline has two distinct stages, each with its own network:

**1. Representation network** -- one CNN (visual/auditory) or GRU-based sequence
model (kinetic) per modality, trained on Study 1 exploratory-action data
(`data/exploratory_action_data/plays_and_options.csv`) to map a raw stimulus (image,
spectrogram, or motion trajectory) to a fixed-size embedding. This is what
`training_scripts/train_*.py` trains, and what the `contrastive` / `autoencoder` /
`VAE` / hybrid / `random` embedding types named in [Running Individual
Experiments](#running-individual-experiments) refer to. `generate_embeddings.py` then
runs every stimulus through the trained
representation network once and caches the results to `results/embeds/`, so nothing
downstream needs the representation network itself again.

**2. Reward network** -- a small MLP (`clea/models/reward.py`) trained on Study 2
human preference data (`data/ranking_data/`) to predict which of two *embeddings* a
person preferred. This is not part of the representation model -- it's the evaluation
mechanism: a good representation should let a reward network learn a person's
preferences from fewer queries. There are two reward-learning approaches, corresponding
to the two result tables:

- **NN-based** (`generate_nn_results.py` -> `results/nn_results.csv`): trains a
  `RewardLearner` MLP per participant via gradient descent, measures held-out
  preference accuracy. Backs the completeness result (Fig. 5).
- **Linear/Bayesian** (`generate_linear_results.py` -> `results/linear_results.csv`):
  uses the `irlpreference` package to simulate a Bayesian linear reward learner
  receiving sequential queries, measuring alignment with the participant's true
  preference vector after each query. Backs the simplicity/minimality results
  (Fig. 6 / Table I).

## Data Loaders

- **`clea/dataloaders/exploratory_loaders.py`** (`RawChoiceDataset`) -- used during
  representation training. Loads raw stimuli (images, spectrograms, or motion
  trajectories) and returns `(anchor, positive, negative)` triplets sampled from a
  participant's Study 1 exploratory-action choices, for contrastive / AE / VAE
  training.
- **`clea/dataloaders/query_loaders.py`** (`UserStudyQueryDataloader`) -- used during
  reward-network training/evaluation. Loads *pairs of precomputed embeddings* (not raw
  stimuli) from a Study 2 preference query and a binary label for which one the
  participant preferred.

## Model Definitions

Representation-network architectures live in `clea/models/`, one file per modality,
each defining a plain encoder plus AE/VAE variants that reuse it:

- **`clea/models/visual.py`** -- `RawImageEncoder`/`RawImageAE`/`RawImageVAE`, a
  convolutional encoder/decoder over `(3, 512, 512)` image frames.
- **`clea/models/auditory.py`** -- `RawAudioEncoder`/`RawAudioAE`/`RawAudioVAE`, the
  same convolutional architecture applied to `(3, 128, 862)` spectrogram images.
- **`clea/models/kinetic.py`** -- `RawSequenceEncoder`/`Seq2Seq`/`Seq2SeqVAE`, a
  bidirectional GRU encoder (plus GRU decoder for the AE/VAE variants) over `(T, 3)`
  motion trajectories.
- **`clea/models/reward.py`** -- `RewardLearner`, the MLP reward network described
  above (Representation Learning Structure).

Which class gets instantiated for a given run is decided by `get_model_and_loss()` at
the top of each `training_scripts/train_*.py`, keyed on the `model_type` /
`embedding_type` (see [Running Individual Experiments](#running-individual-experiments)).

## Loss Functions

- **`contrastive`**: `nn.TripletMarginLoss` directly on encoder output (no
  reconstruction) -- pulls an anchor's embedding toward the embedding of another
  stimulus chosen alongside it and away from a rejected one.
- **`autoencoder`**: `nn.MSELoss` reconstruction loss on anchor/positive/negative,
  plus an L2-norm penalty on the embeddings (`clea/utils/train_utils.py`).
- **`VAE`**: reconstruction MSE + a KL-divergence term against a standard normal
  prior, weighted by `beta`. Implemented as each model class's own `vae_loss()`
  method (`RawImageVAE.vae_loss`, `RawAudioVAE.vae_loss`, `Seq2SeqVAE.vae_loss`).
- **`contrastive+autoencoder`** / **`contrastive+VAE`**: the AE/VAE loss above plus a
  `nn.TripletMarginLoss` term computed on the same model's encoded output, so both a
  reconstruction and a preference-triplet signal shape the embedding.
- **`random`**: no loss -- the encoder is left at its random initialization, used as
  a baseline.
- **Reward network (NN-based, `generate_nn_results.py`)**: `nn.CrossEntropyLoss` over
  the two candidate rewards, plus an L1 penalty on each reward output.
- **Reward network (linear/Bayesian, `generate_linear_results.py`)**: not a gradient-
  descent loss -- `irlpreference`'s `LuceShepardChoice` input model and
  `MonteCarloLinearReward` posterior are updated via Bayesian inference after each
  query, and alignment is measured as cosine similarity to the true preference vector.

## Running Individual Experiments

An "experiment" is one representation network trained for a given (modality, signal,
embedding type, embedding dimensionality) combination, then evaluated by both reward
networks. The full sweep is 3 modalities x 4 signals x 6 embedding types x 5
dimensions. `train_all.py` / `train_config.py` run that whole sweep by default; each
stage below can also be run (and configured) individually if you only want a subset.

Trained models and embeddings follow this naming scheme:

```
{modality}&{task_dependency}&{pretraining}&{embedding_type}&{signal}&{dim}.pth
```

For example: `visual&independent&raw&contrastive&idle&128.pth`

- **modality**: `visual`, `auditory`, `kinetic`
- **task_dependency**: `independent` (one model per signal)
- **pretraining**: `raw` (trained from scratch)
- **embedding_type**: `contrastive`, `autoencoder`, `VAE`, `contrastive+autoencoder`, `contrastive+VAE`, `random`
- **signal**: `idle`, `searching`, `has_item`, `has_information`
- **dim**: embedding dimensionality (e.g. `8`, `16`, `32`, `64`, `128`)

### Training

**All modalities at once (recommended):** `train_all.py` trains visual, auditory, and
kinetic models in one run, reading hyperparameters from `train_config.py`. Every
training script reads that same file, so it is the one place to edit:

```bash
python training_scripts/train_all.py

# or train a subset of modalities:
python training_scripts/train_all.py --modalities visual auditory
```

**One modality at a time:** each script can still be run on its own, and also reads
its hyperparameters from `train_config.py` (the shared block plus that modality's
override dict):

```bash
python training_scripts/train_visual.py
python training_scripts/train_auditory.py
python training_scripts/train_kinetic.py
```

By default, both paths sweep `EMBEDDING_DIMS = [8, 16, 32, 64, 128]` and all six
`MODEL_TYPES` (`contrastive` = CLEA, `autoencoder`, `VAE`, `contrastive+autoencoder`
= CLEA+AE, `contrastive+VAE` = CLEA+VAE, `random`), producing every checkpoint
needed for both the completeness (Fig. 5, dim=128 only) and simplicity/minimality
(Table I / Fig. 6, all five dims) results. The default `margin`/`beta` values in
`train_config.py` are the per-modality hyperparameters used to produce the paper's
reported numbers
(margin=0.1 for visual/auditory, margin=2.0 for kinetic; beta=1.0 for visual,
beta=10.0 for auditory/kinetic).

### Evaluation

**All at once (recommended):** after training, `evaluate_all.py` runs the full
evaluation pipeline in one command: generates embeddings from
`results/trained_models`, then both result tables from those embeddings.

```bash
python evaluation_scripts/evaluate_all.py
```

Equivalent to running these three individually, in order:

```bash
# 1. results/trained_models -> results/embeds
python evaluation_scripts/generate_embeddings.py

# 2. results/embeds -> results/nn_results.csv (recoverability)
python evaluation_scripts/generate_nn_results.py

# 3. results/embeds -> results/linear_results.csv (simplicity / minimality)
python evaluation_scripts/generate_linear_results.py
```

### Plotting

By default these read from `results/` (the output of the Evaluation step above, using
your own retrained models). Pass `--pregenerated` to any of them to instead read from
the committed `pregenerated_results/` snapshot and reproduce the paper's
figures/table without training or generating anything first. Each script saves its
figure as a PNG under `<results_dir>/plots/` (i.e. `results/plots/` or
`pregenerated_results/plots/`, matching whichever data source it read from), in
addition to the interactive `plt.show()`:

```bash
# Fig. 5 -- Completeness: NN-based preference accuracy
python evaluation_scripts/eval_recoverability.py [--pregenerated]

# Fig. 6 -- Minimality: linear preference alignment curves (dim=8)
python evaluation_scripts/eval_simplicity.py [--pregenerated]

# Table I -- Simplicity: alignment AUC vs. embedding dimensionality
python evaluation_scripts/eval_minimality.py [--pregenerated]

# Fig. 7 -- Explainability: nearest-exemplar interpretability
python evaluation_scripts/eval_nn_interpretability.py [--pregenerated]

# Significance testing backing the "(all p < .05)" / asterisk annotations
# in Fig. 5 and Table I
python evaluation_scripts/eval_stats.py [--pregenerated]
```
