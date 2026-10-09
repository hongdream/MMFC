import csv
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset, TensorDataset
from model import check_finite
from preprocessing import TextCorpusPreprocessor
from data_cache import load_processed_cache, save_processed_cache

LOGGER = logging.getLogger("stfed")
DATASETS = {"libri1024": "1024-LibriSpeech", "tedlium1024": "1024-TEDLIUM_release3"}


@dataclass(frozen=True)
class PairedData:
    dataset: TensorDataset
    vocabulary: list[str]
    text_dim: int
    speech_dim: int
    cache_hit: bool = False


def read_paired_inputs(args):
    csv.field_size_limit(2**31 - 1)
    with open(args.data_csv, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"id", "text", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain id, text and split columns")
        rows = [row for row in reader if row["split"] == args.split]
    if not rows or any((not row["text"].strip() for row in rows)):
        raise ValueError("The selected split is empty or contains empty text")
    with np.load(args.boaw_path, allow_pickle=True) as archive:
        speech = np.asarray(archive[args.split], dtype=np.float32)
        speech_ids = [str(value) for value in archive[f"{args.split}_ids"]]
    text_ids = [row["id"] for row in rows]
    if speech.ndim != 2 or len(speech) != len(rows):
        raise ValueError("Text and speech sample counts or shapes differ")
    if len(speech_ids) != len(speech) or len(set(speech_ids)) != len(speech_ids):
        raise ValueError("BoAW IDs must be unique and match its row count")
    if len(set(text_ids)) != len(rows) or set(text_ids) != set(speech_ids):
        raise ValueError("CSV and BoAW IDs differ or are duplicated")
    lookup = {key: index for index, key in enumerate(speech_ids)}
    return ([row["text"] for row in rows], speech[[lookup[key] for key in text_ids]])


def load_data(args):
    cached, cache_path = load_processed_cache(args)
    if cached is not None:
        text, speech, vocabulary = cached
        cache_hit = True
    else:
        import nltk

        nltk.data.path.insert(0, str(Path(__file__).parent / ".nltk_data"))
        documents, speech = read_paired_inputs(args)
        processor = TextCorpusPreprocessor(documents)
        processor.lemmatized_documents = processor._lemmatize_batch(documents)
        processor.build_bag_of_words(
            processor.lemmatized_documents, args.min_token_frequency, args.max_token_frequency_ratio
        )
        vocabulary = [processor.id_to_token[index] for index in range(len(processor.id_to_token))]
        text = np.asarray(processor.bag_of_words, dtype=np.float32)
        cache_hit = False
    validate_paired_features(text, speech, 1024)
    if len(vocabulary) != text.shape[1]:
        raise ValueError("Vocabulary and feature dimensions differ")
    if not cache_hit:
        save_processed_cache(args, text, speech, vocabulary)
    LOGGER.info(
        "[DATA] dataset=%s | source=%s | samples=%d | BoW=%d | BoAW=%d",
        args.dataset,
        "cache" if cache_hit else "preprocessed",
        len(text),
        text.shape[1],
        speech.shape[1],
    )
    return PairedData(
        TensorDataset(torch.from_numpy(text), torch.from_numpy(speech)),
        vocabulary,
        text.shape[1],
        speech.shape[1],
        cache_hit,
    )


def validate_paired_features(text, speech, expected_boaw_dim=None):
    for array, name in ((text, "text BoW"), (speech, "speech BoAW")):
        if array.ndim != 2 or min(array.shape) == 0:
            raise ValueError(f"{name}: expected a nonempty 2D matrix, got {array.shape}")
        check_finite(torch.as_tensor(array), name)
    if len(text) != len(speech):
        raise ValueError("Text and speech sample counts differ")
    if (text < 0).any() or (speech < 0).any():
        raise ValueError("BoW and BoAW must be nonnegative")
    if expected_boaw_dim is not None and speech.shape[1] != expected_boaw_dim:
        raise ValueError(
            f"Selected BoAW dimension is {expected_boaw_dim}, but the archive contains {speech.shape[1]} features"
        )


def build_partition_features(dataset):
    if not isinstance(dataset, TensorDataset) or len(dataset.tensors) != 2:
        raise TypeError("K-means partition expects a TensorDataset of (BoW, BoAW)")
    arrays = []
    for tensor, name in zip(dataset.tensors, ("text BoW", "speech BoAW")):
        value = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if value.ndim != 2 or len(value) != len(dataset):
            raise ValueError(f"{name}: invalid partition matrix shape {value.shape}")
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        arrays.append(np.divide(value, norm, out=np.zeros_like(value), where=norm > 0))
    return np.concatenate(arrays, axis=1) / np.sqrt(2.0)


def kmeans_partition_indices(dataset, num_clients, seed, minimum):
    if num_clients < 1 or minimum < 1 or len(dataset) < num_clients * minimum:
        raise ValueError("Not enough samples for the requested clients and clusters")
    features = build_partition_features(dataset)
    LOGGER.info(
        "[PARTITION] K-means fitting | features=%s | n_clusters=%d | n_init=10",
        tuple(features.shape),
        num_clients,
    )
    fitted = KMeans(n_clusters=num_clients, init="k-means++", n_init=10, random_state=seed).fit(
        features
    )
    assignment = fitted.labels_.copy()
    counts = np.bincount(assignment, minlength=num_clients)
    if minimum:
        centers = fitted.cluster_centers_
        for target in np.argsort(counts):
            while counts[target] < minimum:
                candidates = np.flatnonzero(counts[assignment] > minimum)
                if not len(candidates):
                    raise RuntimeError("Unable to satisfy the minimum K-means client size")
                distances = np.square(features[candidates] - centers[target]).sum(axis=1)
                index = candidates[np.argmin(distances)]
                donor = assignment[index]
                assignment[index] = target
                counts[donor] -= 1
                counts[target] += 1
    rng = np.random.default_rng(seed)
    parts = [rng.permutation(np.flatnonzero(assignment == client)) for client in range(num_clients)]
    combined = np.concatenate(parts)
    if len(combined) != len(dataset) or len(np.unique(combined)) != len(dataset):
        raise RuntimeError("K-means partition must assign every paired sample exactly once")
    LOGGER.info(
        "[PARTITION] K-means assignment complete | client_sizes=%s", [len(part) for part in parts]
    )
    return parts


def make_loaders(dataset, parts, batch_size):
    return [
        DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        for indices in parts
    ]
