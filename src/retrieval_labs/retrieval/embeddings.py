"""Text embeddings behind a small interface, with a disk cache.

Only this module and ``reranker.py`` import sentence-transformers (enforced by tests), and
they import it lazily so the rest of the package works without loading torch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str  # exact Hugging Face commit, for reproducibility
    query_prefix: str = ""

    @property
    def name(self) -> str:
        return f"{self.repo}@{self.revision[:8]}"


# Verified 2026-10-20 on the Hugging Face API: MIT, 33.4M parameters, 512-token input.
BGE_SMALL = ModelSpec(
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    # The model card's documented instruction for retrieval queries (passages get none).
    query_prefix="Represent this sentence for searching relevant passages: ",
)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalized float32 matrix of shape (len(texts), dimension)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """L2-normalized float32 vector."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, spec: ModelSpec = BGE_SMALL, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.name = spec.name
        self.batch_size = batch_size
        self._model = SentenceTransformer(spec.repo, revision=spec.revision, device="cpu")
        self.max_seq_length: int = self._model.max_seq_length

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + text])[0]

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """Tokens per text including special tokens, to measure truncation."""
        encoded = self._model.tokenizer(list(texts), add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]


class CachedEmbedder:
    """Caches document matrices on disk, keyed by model and the exact texts.

    One file per call (in practice, per contract and chunking), so cached vectors are
    reused across runs and any change in chunk text produces a different key.
    """

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self._dir = cache_dir / hashlib.sha256(inner.name.encode("utf-8")).hexdigest()[:16]
        self._queries: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def _path(self, texts: Sequence[str]) -> Path:
        digest = hashlib.sha256()
        for text in texts:
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return self._dir / f"{digest.hexdigest()}.npy"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        path = self._path(texts)
        if path.is_file():
            vectors = np.load(path)
            if vectors.shape[0] == len(texts):
                self.hits += 1
                return vectors
        self.misses += 1
        vectors = self.inner.embed_documents(texts)
        self._dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.stem + ".partial.npy")
        np.save(partial, vectors)
        partial.replace(path)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._queries:
            self._queries[text] = self.inner.embed_query(text)
        return self._queries[text]
