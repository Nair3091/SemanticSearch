"""
Track 2 — Vision & Retrieval
Owner: Dhruv Nair

Two-pass adaptive sampling:
  Pass 1: coarse uniform sampling across the full video (cheap, low-res coverage)
  Pass 2: dense re-sampling inside windows that scored well in Pass 1 (expensive,
          precise, but restricted to a small fraction of the video)

Design notes:
- Pass 1 gives you a coarse similarity curve over time. Window selection
  (deciding WHICH parts of that curve are worth a second look) lives here
  too, since it's a sampling decision — it just consumes scores computed
  elsewhere (CLIP scoring, in clip_encoder.py). Keeps scoring logic and
  sampling logic in separate files even though they're used together.
"""

from typing import List, Tuple
import cv2
import numpy as np

from src.common.types import Frame


def coarse_sample(video_path: str, interval_seconds: float) -> List[Frame]:
    """
    Pass 1: Uniformly sample frames from the full video at a fixed interval.

    Opens the video once, computes timestamps at the given interval, and
    seeks/reads frames at each timestamp. Frames are returned as RGB
    numpy arrays (CLIP expects RGB, OpenCV decodes as BGR by default).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    frames: List[Frame] = []
    timestamp = 0.0
    while timestamp < duration:
        frame_index = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr_frame = cap.read()
        if not ok:
            break
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frames.append(Frame(timestamp=timestamp, frame_index=frame_index, image=rgb_frame))
        timestamp += interval_seconds

    cap.release()
    return frames


def dense_resample(video_path: str, window_start: float, window_end: float,
                    interval_seconds: float) -> List[Frame]:
    """
    Pass 2: Densely re-sample frames within a specific time window
    identified as promising by Pass 1 scoring.

    Same decode logic as coarse_sample, but restricted to [window_start, window_end]
    and at a much finer interval — this is what lets you pin down precise
    moment boundaries without paying the cost of dense-sampling the whole video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames: List[Frame] = []
    timestamp = window_start
    while timestamp <= window_end:
        frame_index = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr_frame = cap.read()
        if not ok:
            break
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frames.append(Frame(timestamp=timestamp, frame_index=frame_index, image=rgb_frame))
        timestamp += interval_seconds

    cap.release()
    return frames


def select_promising_windows(
    frames: List[Frame],
    scores: np.ndarray,
    top_k: int,
    window_radius_seconds: float,
) -> List[Tuple[float, float]]:
    """
    Given Pass-1 frames and their fused similarity scores, pick the top_k
    highest-scoring timestamps, expand each into a window of
    ±window_radius_seconds, and merge any windows that overlap.

    scores: 1D array, same length and order as `frames`, one fused score
            per frame (already aggregated across sub-queries — this function
            doesn't care how that fusion happened).

    Returns a list of (window_start, window_end) tuples, sorted by start time,
    ready to be passed to dense_resample().
    """
    if len(frames) == 0:
        return []
    if len(frames) != len(scores):
        raise ValueError("frames and scores must be the same length")

    k = min(top_k, len(frames))
    # indices of the k highest scores, highest first
    top_indices = np.argsort(scores)[::-1][:k]

    raw_windows = []
    for idx in top_indices:
        center = frames[idx].timestamp
        start = max(0.0, center - window_radius_seconds)
        end = center + window_radius_seconds
        raw_windows.append((start, end))

    raw_windows.sort(key=lambda w: w[0])

    merged: List[Tuple[float, float]] = []
    for start, end in raw_windows:
        if merged and start <= merged[-1][1]:
            # overlaps the previous window — merge instead of duplicating work
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
