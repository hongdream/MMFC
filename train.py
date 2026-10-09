import argparse
import copy
import csv
import logging
import random
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from torch.utils.data import DataLoader
from model import build_spherical_vae_client, check_mixture
from data import DATASETS, load_data, kmeans_partition_indices, make_loaders
from federated import (
    broadcast,
    spherical_vae_client_round,
    collect_z,
    initialize_mixture,
    update_global_mixture,
)

LOGGER = logging.getLogger("stfed")


def cluster_weight_for_round(round_index, target_weight, warmup_rounds):
    if round_index < 1:
        raise ValueError("round_index must start from 1")
    if not np.isfinite(target_weight) or target_weight < 0:
        raise ValueError("target_weight must be finite and nonnegative")
    if warmup_rounds < 0:
        raise ValueError("warmup_rounds must be nonnegative")
    if warmup_rounds <= 1:
        return float(target_weight)
    progress = min((round_index - 1) / (warmup_rounds - 1), 1.0)
    return float(target_weight * progress)


@torch.no_grad()
def initialize_posterior_concentration(server, target_kappa):
    if not np.isfinite(target_kappa) or target_kappa <= 0:
        raise ValueError("target_kappa must be finite and positive")
    encoder = server.shared_model.encoder
    raw_kappa = target_kappa - encoder.concentration_floor
    if raw_kappa <= 0:
        raise ValueError("target_kappa must exceed the concentration floor")
    raw = encoder.concentration_head.bias.new_tensor(raw_kappa)
    inverse_softplus = raw + torch.log(-torch.expm1(-raw))
    encoder.concentration_head.weight.zero_()
    encoder.concentration_head.bias.fill_(inverse_softplus.item())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


@torch.no_grad()
def evaluate(server, loader, device):
    LOGGER.info("[EVAL] encoding the complete selected split")
    server.eval()
    directions, concentrations, predictions, log_likelihoods = ([], [], [], [])
    for text, speech in loader:
        posterior = server.encode_posterior(text.to(device), speech.to(device))
        mu, kappa = (posterior["mu"], posterior["kappa"])
        directions.append(mu.cpu())
        concentrations.append(kappa.cpu())
        predictions.append(server.vmfmm.predict(mu))
        log_likelihoods.append(server.vmfmm.log_prob(mu).cpu())
    directions = torch.cat(directions).numpy()
    concentrations = torch.cat(concentrations).numpy()
    prediction = np.concatenate(predictions)
    counts = np.bincount(prediction, minlength=server.vmfmm.n_cluster)
    proportions = counts[counts > 0] / len(prediction)
    num_predicted_clusters = len(np.unique(prediction))
    direction_norms = np.linalg.norm(directions, axis=1)
    evaluation_log = (
        LOGGER.info if num_predicted_clusters == server.vmfmm.n_cluster else LOGGER.warning
    )
    evaluation_log(
        "[EVAL] cluster_usage=%s | samples=%d | predicted_clusters=%d/%d | counts=%s | mu_norm=[%.6f, %.6f]",
        "PASS" if num_predicted_clusters == server.vmfmm.n_cluster else "CHECK",
        len(prediction),
        num_predicted_clusters,
        server.vmfmm.n_cluster,
        counts.tolist(),
        direction_norms.min(),
        direction_norms.max(),
    )
    cluster_entropy = float(-(proportions * np.log(proportions)).sum())
    metrics = dict(
        PredictedClusters=num_predicted_clusters,
        ClusterEntropy=cluster_entropy,
        NormalizedClusterEntropy=(
            cluster_entropy / np.log(server.vmfmm.n_cluster) if server.vmfmm.n_cluster > 1 else 1.0
        ),
        MinClusterFraction=float(counts.min() / len(prediction)),
        MaxClusterFraction=float(counts.max() / len(prediction)),
        VMFNLL=float(-torch.cat(log_likelihoods).mean()),
        PosteriorKappaMin=float(np.min(concentrations)),
        PosteriorKappaMedian=float(np.median(concentrations)),
        PosteriorKappaMax=float(np.max(concentrations)),
    )
    if 1 < num_predicted_clusters < len(prediction):
        metrics.update(
            Silhouette=silhouette_score(directions, prediction, metric="cosine"),
            CalinskiHarabasz=calinski_harabasz_score(directions, prediction),
            DaviesBouldin=davies_bouldin_score(directions, prediction),
        )
    else:
        LOGGER.warning(
            "[EVAL] geometric metrics unavailable: predicted %d cluster(s)", num_predicted_clusters
        )
    return (metrics, prediction)


def save_models(server, vocabulary, parts, args):
    config = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
    }
    shared = {key: value.detach().cpu() for key, value in server.shared_model.state_dict().items()}
    torch.save(
        dict(
            state_dict=shared,
            config=config,
            vocabulary=vocabulary,
            client_indices=[part.tolist() for part in parts],
        ),
        args.output_dir / "server_model.pt",
    )
    torch.save(
        dict(
            state_dict={
                key: value.detach().cpu() for key, value in server.vmfmm.state_dict().items()
            },
            n_clusters=args.n_clusters,
            latent_dim=args.latent_dim,
        ),
        args.output_dir / "global_vmfmm.pt",
    )
    LOGGER.info(
        "[CHECKPOINT] saved server_model.pt and global_vmfmm.pt to %s", args.output_dir.resolve()
    )


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def log_metrics(stage, metrics):
    values = " | ".join((f"{name}={value:.6g}" for name, value in metrics.items()))
    LOGGER.info("%s | %s", stage, values)


def run_selected_datasets(args):
    names = list(DATASETS) if args.dataset == "both" else [args.dataset]
    for name in names:
        selected = copy.deepcopy(args)
        selected.dataset = name
        directory = selected.data_root / DATASETS[name]
        selected.data_csv = str(directory / "all_data.csv")
        selected.boaw_path = str(directory / "boaw_aligned.npz")
        selected.output_dir = args.output_dir / name
        run(selected)


def parse_args():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="MMFC simplified public training release",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--data_root", type=Path, default=base / "datasets")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output_dir", type=Path, default=base / "outputs")
    parser.add_argument("--data_cache_dir", type=Path, default=base / ".data_cache")
    parser.add_argument("--rebuild_data_cache", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    integer_defaults = dict(
        num_clients=5,
        rounds=50,
        pretrain_rounds=20,
        batch_size=128,
        vmf_init_iters=100,
        local_epochs=1,
        latent_dim=20,
        decoder_hidden=256,
        n_clusters=6,
        seed=42,
        cluster_warmup_rounds=15,
        min_token_frequency=10,
    )
    for name, value in integer_defaults.items():
        parser.add_argument(f"--{name}", type=int, default=value)
    parser.add_argument("--encoder_layers", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--decoder_layers", type=int, nargs="+", default=None)
    float_defaults = dict(
        lr=0.002,
        client_lr=2e-05,
        lambda_cluster=0.001,
        lambda_text=1.0,
        lambda_speech=1.0,
        posterior_kappa_init=200.0,
        max_token_frequency_ratio=0.05,
    )
    for name, value in float_defaults.items():
        parser.add_argument(f"--{name}", type=float, default=value)
    args = parser.parse_args()
    for name in integer_defaults:
        if (
            name not in ("rounds", "pretrain_rounds", "seed", "cluster_warmup_rounds")
            and getattr(args, name) < 1
        ):
            parser.error(f"--{name} must be positive")
    for name in ("rounds", "pretrain_rounds", "seed", "cluster_warmup_rounds"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if args.latent_dim < 2 or args.n_clusters < 2:
        parser.error("Need latent_dim and n_clusters >= 2")
    args.encoder_hidden = args.decoder_hidden
    args.decoder_layers = args.decoder_layers or list(reversed(args.encoder_layers))
    if any((value < 1 for value in args.encoder_layers + args.decoder_layers)):
        parser.error("Layer multipliers must be positive")
    for name in float_defaults:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    for name in ("lr", "client_lr", "lambda_text", "lambda_speech", "max_token_frequency_ratio"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_token_frequency_ratio > 1:
        parser.error("--max_token_frequency_ratio must not exceed 1")
    try:
        args.device = torch.device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu")
        if args.device.index is not None and args.device.index >= torch.cuda.device_count():
            parser.error("Requested CUDA device is unavailable")
    args.implementation = "simplified"
    return args


def initialize_all(server, clients, loaders, sizes, args):
    broadcast(server, clients)
    for index, (client, loader) in enumerate(zip(clients, loaders)):
        directions = collect_z(client, loader, args.device)
        initialize_mixture(client.vmfmm, directions, args.vmf_init_iters)
        LOGGER.info("[INIT] client=%d | samples=%d | clustering=KMeans", index + 1, sizes[index])
    update_global_mixture(server, clients, args.vmf_init_iters)


def run(args):
    started = perf_counter()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    LOGGER.info(
        "[RUN] implementation=simplified | dataset=%s | seed=%d | device=%s",
        args.dataset,
        args.seed,
        args.device,
    )
    paired = load_data(args)
    dataset = paired.dataset
    parts = kmeans_partition_indices(dataset, args.num_clients, args.seed, args.n_clusters)
    loaders = make_loaders(dataset, parts, args.batch_size)
    sizes = list(map(len, parts))
    evaluation_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    server = build_spherical_vae_client(
        paired.text_dim,
        paired.speech_dim,
        args.latent_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        n_clusters=args.n_clusters,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
    ).to(args.device)
    clients = [copy.deepcopy(server) for _ in parts]
    history = []
    for index in range(args.pretrain_rounds):
        loss = spherical_vae_client_round(
            server, clients, loaders, sizes, args, lambda_cluster=0.0, sample=False
        )
        LOGGER.info("[PRETRAIN] round=%d/%d | loss=%.6f", index + 1, args.pretrain_rounds, loss)
    if args.posterior_kappa_init > 0:
        initialize_posterior_concentration(server, args.posterior_kappa_init)
    initialize_all(server, clients, loaders, sizes, args)
    metrics, prediction = evaluate(server, evaluation_loader, args.device)
    history.append(dict(round=0, **metrics))
    log_metrics("[INITIALIZED]", metrics)
    for index in range(1, args.rounds + 1):
        cluster_weight = cluster_weight_for_round(
            index, args.lambda_cluster, args.cluster_warmup_rounds
        )
        sample = index > args.cluster_warmup_rounds
        client_loss = spherical_vae_client_round(
            server, clients, loaders, sizes, args, cluster_weight, sample=sample, lr=args.client_lr
        )
        update_global_mixture(server, clients, args.vmf_init_iters)
        metrics, prediction = evaluate(server, evaluation_loader, args.device)
        history.append(
            dict(
                round=index,
                client_loss=client_loss,
                cluster_weight=cluster_weight,
                posterior_mode="vMF" if sample else "mu",
                **metrics,
            )
        )
        log_metrics(f"[ROUND {index}/{args.rounds}]", metrics)
    check_mixture(server.vmfmm, "final global mixture")
    save_models(server, paired.vocabulary, parts, args)
    np.save(args.output_dir / "predictions.npy", prediction.astype(np.int64))
    fields = list(dict.fromkeys((key for row in history for key in row)))
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(history)
    log_metrics("[FINAL]", metrics)
    LOGGER.info(
        "[RUN] COMPLETE | output=%s | elapsed=%.1fs", args.output_dir, perf_counter() - started
    )


if __name__ == "__main__":
    try:
        run_selected_datasets(parse_args())
    except Exception:
        LOGGER.exception("Training failed")
        raise
