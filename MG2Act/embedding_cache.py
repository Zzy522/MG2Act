"""Versioned, optional on-disk cache of frozen ESM-C outputs (never projections)."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import warnings

import torch


CACHE_SCHEMA = 1


def backbone_fingerprint(model, tokenizer, settings):
    """Hash the actual loaded weights, including weights restored from a checkpoint."""
    metadata = {
        "schema": CACHE_SCHEMA,
        "model": "esmc_300m",
        "esm_version": importlib.metadata.version("esm"),
        "torch_version": str(torch.__version__),
        "tokenizer": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
        "vocabulary": tokenizer.get_vocab(),
        "architecture": repr(model),
        "settings": settings,
        "special_tokens": "tokenizer.encode: retain BOS/EOS; no truncation",
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode("utf-8"))
        # uint8 supports all parameter dtypes, including bfloat16.
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


class EmbeddingDiskCache:
    def __init__(self, root, fingerprint):
        self.root = Path(root) / fingerprint
        self.fingerprint = fingerprint

    def _path(self, sequence):
        key = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        return self.root / (key + ".pt")

    def load(self, sequence, length, width):
        path = self._path(sequence)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            embedding = payload["embedding"]
            if (payload["schema"] != CACHE_SCHEMA
                    or payload["fingerprint"] != self.fingerprint
                    or payload["sequence"] != sequence
                    or not isinstance(embedding, torch.Tensor)
                    or embedding.shape not in ((length, width), (1, length, width))
                    or not embedding.is_floating_point()
                    or not torch.isfinite(embedding).all()):
                raise ValueError("cache metadata or tensor shape/content mismatch")
            return embedding.detach().clone()
        except Exception as exc:
            warnings.warn(f"Ignoring invalid ESM-C embedding cache {path}: {exc}", RuntimeWarning)
            return None

    def save(self, sequence, embedding):
        temporary = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                torch.save({
                    "schema": CACHE_SCHEMA,
                    "fingerprint": self.fingerprint,
                    "sequence": sequence,
                    "embedding": embedding.detach().cpu(),
                }, file)
            os.replace(temporary, self._path(sequence))
        except (OSError, RuntimeError) as exc:
            # A read-only/full disk must not prevent uncached training.
            warnings.warn(f"Could not persist ESM-C embedding cache: {exc}", RuntimeWarning)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
