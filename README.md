# MMFC

Trainable simplified public release for multimodal federated clustering.

## Version

This release retains the paired text/speech spherical VAE, local reconstruction training, a basic cluster negative log likelihood, and sample-count-weighted FedAvg. It omits selected components of the complete method while the paper is under review.

Private-mixture-weighted latent sampling, source-client pseudo-feature decoding, server pseudo-reconstruction distillation, latent consistency loss, and posterior-to-mixture KL are not included. Mixture initialization uses ordinary K-means with fixed initial concentrations instead of iterative vMF EM. The global mixture is refitted from uniformly pooled client centers. Training saves the final round.

Training and inference work in this version. Its metrics describe this simplified algorithm and should not be treated as a reproduction of the complete paper results. Performance may differ; no performance comparison is claimed here. The package contains no hidden complete implementation, disabled placeholder functions, experiment scripts, or code comments.

## Installation

Use Python 3.11 or later. From this directory:

```sh
python -m pip install -r requirements.txt
python -m nltk.downloader -d .nltk_data punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords wordnet omw-1.4
```

The default device is `cuda:0` when available and otherwise the CPU. Select a device using `--device cpu` or another available CUDA device.

## Datasets

The package includes processed text and 1,024-dimensional speech bag-of-audio-words (BoAW) features. Raw audio is not included.

| Dataset | Directory | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| LibriSpeech | `datasets/1024-LibriSpeech` | 5,466 | 188 | 177 |
| TED-LIUM Release 3 | `datasets/1024-TEDLIUM_release3` | 2,351 | 8 | 11 |

Each directory contains `all_data.csv`, `boaw_aligned.npz`, `codebook.npy`, `corpus.tsv`, and `vocabulary.txt`. CSV columns are `id`, `text`, and `split`. NPZ arrays contain each split's BoAW features and corresponding IDs. The loader aligns paired rows by sample ID and builds its filtered text bag-of-words vocabulary from the selected split.

## Training

Train both datasets sequentially:

```sh
python train.py
```

Train one dataset:

```sh
python train.py --dataset libri1024
python train.py --dataset tedlium1024
```

Inspect training parameters:

```sh
python train.py --help
```

Defaults are 5 clients, 6 mixture components, a 20-dimensional latent space, 20 deterministic pretraining rounds, and 50 federated rounds. The shared encoder and decoder hidden width is 256. Client cluster loss ramps up during a 15-round warmup, with stochastic posterior reconstruction thereafter. The client objective uses paired reconstruction plus a basic mixture negative log likelihood on posterior directions.

Relative paths work after moving the folder or extracting the archive. `--data_root` can select another directory with the same two dataset subdirectories. Preprocessed features are cached locally; `--rebuild_data_cache` rebuilds them. A selected split needs at least `num_clients * n_clusters` samples.

## Outputs

Each dataset writes `training.log`, `metrics.csv`, `server_model.pt`, `global_vmfmm.pt`, and `predictions.npy` under `outputs/<dataset>`. Checkpoint configuration records `implementation: simplified`. Predictions follow the selected CSV split's row order. Reusing an output directory overwrites files with the same names.

The checkpoints contain shared model weights, global mixture weights, training configuration, text vocabulary, and client sample indices. Reload the shared and global weights into `build_spherical_vae_client` with the saved dimensions and layer settings, then call `predict_cluster` with paired feature tensors.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Training entry point, basic schedule, evaluation, checkpoint saving |
| `networks.py` | Paired encoder/decoder and reconstruction losses |
| `model.py` | Spherical VAE client, basic clustering objective, validation |
| `vmf.py` | Standard vMF sampling/density and simplified mixture fitting |
| `federated.py` | FedAvg, local training, global mixture refitting |
| `data.py` | Paired data loading and K-means client partitioning |
| `data_cache.py` | Local preprocessed feature cache |
| `preprocessing.py` | Text preprocessing and BoW construction |
