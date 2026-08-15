"""
Track 2 — Vision & Retrieval
Owner: Dhruv Nair

Responsible for:
- Building and persisting a FAISS index over precomputed frame embeddings
- Fast top-k similarity lookup at inference time
"""

import numpy as np


def build_index(frame_embeddings: np.ndarray, index_path: str) -> None:
    """
    Build a FAISS index from frame embeddings and save it to disk.
    """
    raise NotImplementedError


def load_index(index_path: str):
    """
    Load a precomputed FAISS index from disk.
    """
    raise NotImplementedError


def search(index, query_embedding: np.ndarray, top_k: int):
    """
    Return the top_k nearest frame embeddings (and their indices/distances)
    for a given query embedding.
    """
    raise NotImplementedError
