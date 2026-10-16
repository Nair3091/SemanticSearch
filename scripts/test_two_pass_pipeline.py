"""
Throwaway test script — NOT part of the orchestrator.
Manually chains Pass 1 -> window selection -> Pass 2 to confirm the
two-pass loop actually works end-to-end, using real CLIP scores.

Run: python scripts/test_two_pass_pipeline.py
"""
import sys
from pathlib import Path
import numpy as np

# Make sure the repo root (parent of scripts/) is on the path, so
# `from src...` imports work no matter how this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision_retrieval.sampling.two_pass_sampler import (
    coarse_sample, dense_resample, select_promising_windows
)
from src.vision_retrieval.embedding.clip_encoder import (
    embed_frames, embed_text, score_frames
)

VIDEO_PATH = "data/raw_videos/sample.mp4" 

QUERY = "a cat"

PASS1_INTERVAL = 2.0
PASS2_INTERVAL = 0.25
TOP_K_WINDOWS = 2
WINDOW_RADIUS_SECONDS = 3.0


def embed_text_ensembled(query: str) -> np.ndarray:
    """
    CLIP scores better with natural photo-caption phrasing than bare labels,
    and even better when you average across several templates (the approach
    from the original CLIP paper's zero-shot classification).
    """
    templates = [
        f"a photo of {query}",
        f"a picture of {query}",
        f"{query}",
        f"a video frame showing {query}",
    ]
    embeddings = embed_text(templates)          # (4, D)
    avg = embeddings.mean(axis=0, keepdims=True)
    return avg / np.linalg.norm(avg, axis=1, keepdims=True)  # re-normalize after averaging


def main():
    print(f"=== PASS 1: coarse sampling every {PASS1_INTERVAL}s ===")
    pass1_frames = coarse_sample(VIDEO_PATH, interval_seconds=PASS1_INTERVAL)
    print(f"Pass 1 frame count: {len(pass1_frames)}")
    print(f"Timestamps: {[round(f.timestamp, 2) for f in pass1_frames]}")

    print("\n=== Scoring Pass 1 frames against query ===")
    frame_embeds = embed_frames(pass1_frames)
    text_embeds = embed_text_ensembled(QUERY)
    scores = score_frames(frame_embeds, text_embeds)  # shape (N, 1)
    scores_1d = scores[:, 0]
    print(f"Pass 1 scores: {[round(s, 4) for s in scores_1d]}")

    print(f"\n=== Selecting top-{TOP_K_WINDOWS} promising windows ===")
    windows = select_promising_windows(
        pass1_frames, scores_1d,
        top_k=TOP_K_WINDOWS,
        window_radius_seconds=WINDOW_RADIUS_SECONDS,
    )
    print(f"Selected windows: {windows}")

    print(f"\n=== PASS 2: dense re-sampling every {PASS2_INTERVAL}s within selected windows ===")
    all_pass2_frames = []
    for start, end in windows:
        dense_frames = dense_resample(VIDEO_PATH, start, end, interval_seconds=PASS2_INTERVAL)
        print(f"Window ({start:.2f}, {end:.2f}) -> {len(dense_frames)} dense frames")
        all_pass2_frames.extend(dense_frames)

    print(f"\nTotal Pass 2 frames: {len(all_pass2_frames)}")

    print("\n=== Re-scoring Pass 2 frames ===")
    dense_embeds = embed_frames(all_pass2_frames)
    dense_scores = score_frames(dense_embeds, text_embeds)[:, 0]

    print("Timestamp -> Score (Pass 2, sorted highest first):")
    ranked = sorted(zip(all_pass2_frames, dense_scores), key=lambda x: -x[1])
    for frame, score in ranked[:10]:
        print(f"  t={frame.timestamp:.2f}s  score={score:.4f}")

    best_frame, best_score = ranked[0]
    print(f"\nBest match: t={best_frame.timestamp:.2f}s, score={best_score:.4f}")


if __name__ == "__main__":
    main()