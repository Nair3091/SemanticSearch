"""
Track 2 — Vision & Retrieval
Owner: Dhruv Nair

Responsible for:
- Embedding frames and text sub-queries into a shared CLIP vector space
- Computing frame-query similarity scores
"""

from typing import List
import numpy as np

from src.vision_retrieval.sampling.two_pass_sampler import Frame


def embed_frames(frames: List[Frame]) -> np.ndarray:
    """
    Run frozen CLIP vision encoder over a batch of frames.
    Returns an (N, D) array of frame embeddings.
    """
    raise NotImplementedError


def embed_text(sub_queries: List[str]) -> np.ndarray:
    """
    Run frozen CLIP text encoder over sub-query strings.
    Returns an (M, D) array of text embeddings.
    """
    raise NotImplementedError


def score_frames(frame_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every frame embedding and every sub-query embedding.
    Returns an (N, M) similarity matrix.
    """
    raise NotImplementedError
