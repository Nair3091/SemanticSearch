"""
Track 2 — Vision & Retrieval
Owner: Dhruv Nair

Responsible for:
- Pass 1: coarse uniform frame sampling across the full video
- Pass 2: dense adaptive re-sampling within promising windows identified in Pass 1
"""

from typing import List

from src.common.types import Frame


def coarse_sample(video_path: str, interval_seconds: float) -> List[Frame]:
    """
    Pass 1: Uniformly sample frames from the full video at a fixed interval.
    """
    raise NotImplementedError


def dense_resample(video_path: str, window_start: float, window_end: float,
                    interval_seconds: float) -> List[Frame]:
    """
    Pass 2: Densely re-sample frames within a specific time window
    identified as promising by Pass 1 scoring.
    """
    raise NotImplementedError
