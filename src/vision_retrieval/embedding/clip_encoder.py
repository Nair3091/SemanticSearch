"""
Track 2 — Vision & Retrieval
Owner: Dhruv Nair

Embeds frames and text sub-queries into a shared CLIP vector space,
and scores frame-query similarity.

Uses Hugging Face `transformers` CLIP (matches the design doc's stated
stack: PyTorch + Hugging Face Transformers). Model name/device come
from configs/config.yaml — don't hardcode them elsewhere.
"""

from typing import List
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.common.types import Frame

_MODEL_NAME = "openai/clip-vit-base-patch32"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded lazily so importing this module doesn't immediately pull the
# model onto the GPU — useful for running tests that don't need it.
_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
        _model.eval()
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
    return _model, _processor


def embed_frames(frames: List[Frame], batch_size: int = 32) -> np.ndarray:
    """
    Run the frozen CLIP vision encoder over a batch of frames.
    Returns an (N, D) array of L2-normalized frame embeddings.

    Frames are processed in mini-batches to avoid blowing up GPU memory
    on long videos with many sampled frames.
    """
    if len(frames) == 0:
        return np.zeros((0, 512), dtype=np.float32)  # 512 = ViT-B/32 embed dim

    model, processor = _load_model()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            images = [f.image for f in batch]
            inputs = processor(images=images, return_tensors="pt").to(_DEVICE)
            output = model.get_image_features(**inputs)
            # transformers 5.x wraps the embedding in .pooler_output;
            # transformers 4.x returns the tensor directly. Handle both.
            features = output.pooler_output if hasattr(output, "pooler_output") else output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def embed_text(sub_queries: List[str]) -> np.ndarray:
    """
    Run the frozen CLIP text encoder over sub-query strings.
    Returns an (M, D) array of L2-normalized text embeddings.
    """
    if len(sub_queries) == 0:
        return np.zeros((0, 512), dtype=np.float32)

    model, processor = _load_model()

    with torch.no_grad():
        inputs = processor(text=sub_queries, return_tensors="pt", padding=True).to(_DEVICE)
        output = model.get_text_features(**inputs)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.cpu().numpy()


def score_frames(frame_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every frame embedding and every sub-query embedding.
    Returns an (N, M) similarity matrix: rows = frames, columns = sub-queries.

    Since both embeddings are already L2-normalized (embed_frames/embed_text
    do this), cosine similarity is just the dot product — no extra normalization
    needed here.
    """
    if frame_embeddings.shape[0] == 0 or text_embeddings.shape[0] == 0:
        return np.zeros((frame_embeddings.shape[0], text_embeddings.shape[0]), dtype=np.float32)

    return frame_embeddings @ text_embeddings.T