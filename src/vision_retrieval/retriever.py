from typing import List
import numpy as np

from src.common.config import load_config
from src.common.types import FrameScores
from src.vision_retrieval.sampling.two_pass_sampler import (
    coarse_sample, dense_resample, select_promising_windows)
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames


def retrieve(video_path: str, sub_queries: List[str]) -> FrameScores:
    if len(sub_queries) == 0:
        raise ValueError("sub_queries must not be empty")
    c = load_config()["vision_retrieval"]

    text_emb = embed_text(sub_queries)
    coarse = coarse_sample(video_path, c["pass1_interval_seconds"])
    coarse_scores = score_frames(embed_frames(coarse), text_emb)       # (N, M)

    # window selection uses min over sub-queries (same rule as the fusion stage)
    windows = select_promising_windows(
        coarse, coarse_scores.min(axis=1) if len(coarse) else np.array([]),
        c["top_k_windows"], c["window_radius_seconds"])

    dense = []
    for s, e in windows:
        dense.extend(dense_resample(video_path, s, e, c["pass2_interval_seconds"]))
    dense_scores = score_frames(embed_frames(dense), text_emb)

    rows = {}
    for f, r in zip(coarse, coarse_scores):
        rows[round(f.timestamp, 3)] = (r, False)
    for f, r in zip(dense, dense_scores):          # dense overwrites coarse at equal timestamps
        rows[round(f.timestamp, 3)] = (r, True)

    keys = sorted(rows)
    S = np.stack([rows[k][0] for k in keys]) if keys else np.zeros((0, len(sub_queries)))
    return FrameScores(
        timestamps=np.array(keys),
        scores=[S[:, j] for j in range(S.shape[1])],
        is_dense=np.array([rows[k][1] for k in keys], dtype=bool),
        windows=windows,
    )