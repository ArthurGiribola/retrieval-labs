"""Regression test for LOG-008: BM25 scores must not depend on Python's hash seed.

Set iteration order of strings changes with PYTHONHASHSEED, and floating-point addition
is not associative, so summing per-term contributions in set order made scores (and the
order of near-tied chunks) differ between processes.
"""

import os
import subprocess
import sys

_PROGRAM = """
import random
from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.bm25 import BM25Retriever

rng = random.Random(0)
vocab = [f"term{i}" for i in range(60)]
chunks, offset = [], 0
for i in range(40):
    text = " ".join(rng.choice(vocab) for _ in range(rng.randint(5, 40)))
    chunks.append(Chunk(chunk_id=f"c#{i:04d}", contract_id="c", index=i, start=offset,
                        end=offset + len(text), text=text, chunking="t"))
    offset += len(text) + 1
index = BM25Retriever().build(chunks)
query = " ".join(vocab[:45])
print([(r.chunk_id, repr(r.score)) for r in index.search(query, 40)])
"""


def run_with_seed(seed: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    result = subprocess.run(
        [sys.executable, "-c", _PROGRAM], capture_output=True, text=True, env=env, check=True
    )
    return result.stdout


def test_scores_are_identical_across_hash_seeds() -> None:
    outputs = {run_with_seed(seed) for seed in range(6)}
    assert len(outputs) == 1
