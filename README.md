# MMFC: Multimodal Federated Clustering

## Intro

This repository contains a trainable, simplified public implementation of MMFC for clustering paired text and speech features. It includes a multimodal spherical variational autoencoder (VAE), local reconstruction training, a basic cluster negative log likelihood, and sample-count-weighted federated averaging (FedAvg).

The release includes processed LibriSpeech and TED-LIUM Release 3 datasets with text and 1,024-dimensional speech bag-of-audio-words (BoAW) features. Raw audio is not included.

> This is a simplified public release while the paper is under review. Training and inference are supported, and its metrics describe this simplified algorithm. The release does not reproduce the complete paper results.

Private-mixture-weighted latent sampling, source-client pseudo-feature decoding, server pseudo-reconstruction distillation, latent consistency loss, and posterior-to-mixture KL are omitted. Mixture initialization uses K-means with fixed initial concentrations, and the global mixture is refitted from uniformly pooled client centers. Training saves the final round.

## Requirements

- Python >= 3.11
- torch >= 2.0
- numpy >= 1.24
- scipy >= 1.10
- scikit-learn >= 1.3
- nltk >= 3.9

Install the dependencies and NLTK resources from the repository root:

```shell
python -m pip install -r requirements.txt
python -m nltk.downloader -d .nltk_data punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords wordnet omw-1.4
```

## How to run

```shell
python train.py --dataset xxx
```

**xxx: select the dataset you want to run — `libri1024`, `tedlium1024`, or `both`.**

| Dataset | Argument | Directory | Train | Validation | Test |
| :--- | :---: | :--- | ---: | ---: | ---: |
| LibriSpeech | `libri1024` | `datasets/1024-LibriSpeech` | 5,466 | 188 | 177 |
| TED-LIUM Release 3 | `tedlium1024` | `datasets/1024-TEDLIUM_release3` | 2,351 | 8 | 11 |

Each dataset directory contains `all_data.csv`, `boaw_aligned.npz`, `codebook.npy`, `corpus.tsv`, and `vocabulary.txt`. CSV columns are `id`, `text`, and `split`; speech features are aligned with text by sample ID.

The default configuration uses 5 clients, 6 clusters, a 20-dimensional latent space, a hidden width of 256, 20 deterministic pretraining rounds, and 50 federated rounds. The cluster loss ramps up during a 15-round warmup.

The default device is `cuda:0` when CUDA is available, otherwise `cpu`. Use `--device cpu` to run on the CPU. View all training parameters with:

```shell
python train.py --help
```

Paths are relative to the repository. Use `--data_root` to select another dataset root and `--rebuild_data_cache` to rebuild cached features. The selected split must contain at least `num_clients * n_clusters` samples.

## Example

Train on LibriSpeech:

```shell
python train.py --dataset libri1024
```

Train on TED-LIUM Release 3:

```shell
python train.py --dataset tedlium1024
```

Train on both datasets sequentially:

```shell
python train.py --dataset both
```

Running `python train.py` also trains on both datasets by default.

Each run saves the following files under `outputs/<dataset>/`:

| File | Description |
| :--- | :--- |
| `training.log` | Training progress and evaluation metrics |
| `metrics.csv` | Per-round evaluation metrics |
| `server_model.pt` | Shared VAE checkpoint |
| `global_vmfmm.pt` | Global von Mises–Fisher mixture checkpoint |
| `predictions.npy` | Cluster assignments in the selected CSV split's row order |

Checkpoints record `implementation: simplified` and include model weights, configuration, text vocabulary, and client sample indices. For inference, load the shared and global weights into `build_spherical_vae_client` using the saved dimensions and layer settings, then call `predict_cluster` with paired feature tensors. Reusing an output directory overwrites files with the same names.

## Comparison Results

No benchmark comparison is reported for this simplified release. The included datasets have no ground-truth class labels, so evaluation uses internal clustering metrics on the selected split (`train` by default).

> **Table 1:** Evaluation metrics produced by this release. These are metric definitions; comparative experiment scores are not provided.

| Metric | Description | Interpretation |
| :--- | :--- | :--- |
| Silhouette | Separation and compactness in the spherical latent space, using cosine distance | Higher is better |
| Calinski–Harabasz | Ratio of between-cluster to within-cluster dispersion | Higher is better for matched data and cluster counts |
| Davies–Bouldin | Similarity between clusters relative to their separation | Lower is better |
| Normalized Cluster Entropy | Balance of cluster sizes | Values closer to 1 indicate more balanced clusters |
| Min / Max Cluster Fraction | Smallest and largest cluster proportions | Diagnose empty or dominant clusters |
| vMF NLL | Negative log likelihood under the global mixture | Lower is better within the same experiment |

Numerical comparison results can be reported from `metrics.csv` after running this release with matched evaluation settings. ACC, NMI, ARI, and Purity require ground-truth labels, which are absent from the supplied datasets.
