import hashlib
import json
import os
from pathlib import Path
import numpy as np


def source_state(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def cache_metadata(args):
    return dict(
        format_version=1,
        data_csv=source_state(args.data_csv),
        boaw_path=source_state(args.boaw_path),
        split=args.split,
        min_token_frequency=args.min_token_frequency,
        max_token_frequency_ratio=args.max_token_frequency_ratio,
    )


def processed_cache_path(args, metadata):
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return (args.data_cache_dir / f"paired_{digest}.npz", encoded)


def load_processed_cache(args):
    path, metadata = processed_cache_path(args, cache_metadata(args))
    if args.rebuild_data_cache or not path.is_file():
        return (None, path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["metadata"].item()) != metadata:
                raise ValueError("Cache metadata differs")
            cached = (
                np.asarray(archive["text"], dtype=np.float32),
                np.asarray(archive["speech"], dtype=np.float32),
                archive["vocabulary"].astype(str).tolist(),
            )
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Unreadable cache: {path}. Use --rebuild_data_cache.") from error
    return (cached, path)


def save_processed_cache(args, text, speech, vocabulary):
    path, metadata = processed_cache_path(args, cache_metadata(args))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            metadata=np.asarray(metadata),
            text=text,
            speech=speech,
            vocabulary=np.asarray(vocabulary, dtype=np.str_),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
