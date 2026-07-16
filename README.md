# DynamicQuantization

Python simulator for **DynQuant**, a per-round, per-direction dynamic weight
quantization strategy for embedded federated learning (FL), as described in
*"Dynamic Weight Quantization for Embedded Federated Learning in LoRa Mesh
Networks: A Case Study"*.

The simulator trains a small feed-forward keyword-spotting model with
federated averaging (FedAvg) across simulated clients, applying different
weight-quantization policies (static, early/late, asymmetric, stochastic) to
the client-to-server and server-to-client communication at each FL round, and
measures the resulting accuracy/communication-cost trade-off.

This repository accompanies the simulation results reported in Section 5 of
the paper. The hardware prototype (Arduino Portenta H7 + TTGO LoRa32 over a
LoRa mesh) is implemented in two companion repositories, linked below.

## Requirements

- Python 3.11+ (uses `X | Y` union type syntax)
- Dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

(`torch`, `torchvision`, `torchaudio`, `matplotlib`, `numpy`, `pandas`, `tqdm`)

A CUDA/MPS accelerator is used automatically if available (`torch.accelerator`),
otherwise the simulator falls back to CPU. All experiments in the paper run
on CPU in a few minutes per experiment.

## Dataset

The experiments use a keyword-spotting dataset of one-second audio
utterances for 4 keywords ("montserrat", "pedraforca", "vermell", "blau"),
180 samples per keyword (720 total), of which 120 are held out as a global
test set. The dataset is not included in this repository (see `.gitignore`);
it is available at
[NilLlisterri/FLLoRaMesher](https://github.com/NilLlisterri/FLLoRaMesher)
(branch `no-reliable-payloads`).

Download it and place it so that the directory structure is:

```
data/keywords/montserrat/*.json
data/keywords/pedraforca/*.json
data/keywords/vermell/*.json
data/keywords/blau/*.json
```

Each `.json` file contains a `payload.values` field with the raw audio
samples for one utterance; features are extracted at load time as 13-band
MFCCs (`KWSDataset.py`).

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | Defines the `Experiment` class (training loop, FedAvg, quantization schedules) and the experiment functions used to produce each figure/table in the paper. |
| `KWSDataset.py` | PyTorch `Dataset` for the keyword-spotting data, with IID / non-IID client partitioning (`iid_factor`). |
| `Nets.py` | Model definitions. `KWSNet` (a 2-layer feed-forward network, `hl_size` neurons in the hidden layer) is the model used in the paper. |
| `quantization.py` | Min-max uniform quantization/dequantization of a flat weight vector to `N` bits. |
| `stc.py` | Sparse ternary compression (STC) baseline: re-implementation of Sattler et al., "Robust and Communication-Efficient Federated Learning From Non-i.i.d. Data" (IEEE TNNLS, 2020) — top-`k` sparsification, ternarization, and the Golomb-optimal communication-cost formula, used for the comparison in Section 5.7. |
| `tests/quantization_tests.py` | Unit tests for `quantization.py`. |
| `requirements.txt` | Python dependencies. |

## Running an experiment

Each experiment in `main.py` is a method of the `Experiment` class that
trains multiple simulated clients over 15 random seeds, plots the result to
`plots/`, and exports the underlying numbers to an `.xlsx` file. Only one
experiment is meant to run at a time; uncomment the relevant line(s) at the
bottom of `main.py` (inside `main()`), then run:

```bash
mkdir -p plots
python main.py
```

Mapping from paper section/figure to the experiment method in `main.py`:

| Paper section / figure | Method |
|---|---|
| Section 5.1, Fig. 2 (accuracy vs. FL round for different static bit widths, incl. "No FL") | `accuracy_vs_epochs_vs_quant_bits_experiment(num_models)` |
| Section 5.2 (early vs. late quantization) | `early_vs_late_quantization(num_models)` |
| Section 5.3 (asymmetric quantization, line-plot exploration) | `asymmetric_quantization_experiment(num_models)` |
| Section 5.4, Fig. 3 (accuracy by hidden-layer size and policy) / Table 4 (transmission cost) | `nn_size_experiment(num_models)` |
| Section 5.5, Fig. 4 (non-IID data) | `iid_vs_non_experiment(num_models)`, `non_iid_policies_experiment(num_models)` |
| Section 5.7, Table 6 (comparison against STC, Sattler et al. 2020) | `stc_sparsity_sweep(num_models, p_values, hl_size=20)` |
| Exploratory (not used for a paper figure) | `accuracy_loss_experiment`, `quantized_weights_histogram_experiment`, `random_quantization_experiment` |

Note: Figure 1 in the paper is the DynQuant design diagram (not produced by
this simulator); the simulator's figures start at Figure 2.

`num_models` is the number of simulated FL clients (3 in the paper).

Key parameters (in `Experiment.__init__`):

- `self.low_bits = 6`, `self.high_bits = 8`: the bit-width range explored by
  the dynamic (early/late/asymmetric/stochastic) policies.
- `self.samples_per_fl_batch = 4`, `self.fl_batches = 50`: 4 local training
  samples per client per round, 50 FL rounds.
- `SEEDS_COUNT = 15`: number of random seeds averaged per curve/bar.
- `hl_size` (passed to `KWSNet`): hidden-layer size in neurons (10, 15, 20,
  or 25 in the paper).

Outputs (figures and `.xlsx` tables) are written to `plots/`, which is
git-ignored and must exist before running (`mkdir -p plots`).

## Reproducibility notes

- All quantization policies (static, early, late, asymmetric, stochastic)
  are schedule-driven functions of the FL round index and are defined inline
  in each experiment method (see the `options`/`cases` lists), not read from
  an external config file.
- Random seeds are drawn once (`self.seeds = random.sample(range(1000), SEEDS_COUNT)`
  in `__init__`) and reused across all quantization policies compared within
  the same experiment, so that per-policy differences are not confounded by
  seed variation.
- The train/test split (600/120 samples) and the non-IID client partitioning
  (`iid_factor` in `KWSDataset`) are deterministic given a seed.

## Related repositories

- Hardware client firmware (Arduino Portenta H7):
  [NilLlisterri/FLLoRaMesher](https://github.com/NilLlisterri/FLLoRaMesher)
  (branch `no-reliable-payloads`)
- LoRa mesh router firmware (TTGO LoRa32):
  [NilLlisterri/TTGO-LoRaMesher](https://github.com/NilLlisterri/TTGO-LoRaMesher)
- LoRa mesh routing library:
  [LoRaMesher/LoRaMesher](https://github.com/LoRaMesher/LoRaMesher)

## Citation

If you use this code, please cite the paper (full reference to be added upon
publication).
