"""
Shared data types used across all three tracks.
Do NOT redefine these inside individual track modules — import from here
so everyone's interfaces stay compatible.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SubQuery:
    text: str
    is_negated: bool


@dataclass
class Frame:
    timestamp: float
    frame_index: int


@dataclass
class CandidateSegment:
    start_time: float
    end_time: float
    fused_confidence: float


@dataclass
class RetrievalResult:
    accepted: bool
    segment: Optional[CandidateSegment]
    message: str  # e.g. "Reject - No result found" or a success message
