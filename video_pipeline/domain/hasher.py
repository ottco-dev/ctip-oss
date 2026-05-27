"""
video_pipeline.domain.hasher — Perceptual hash-based frame deduplication.

Detects near-duplicate frames in video sequences to avoid redundant analysis.
Microscopy videos often have static or nearly-static periods (e.g., focusing,
waiting for stage movement) where consecutive frames are nearly identical.

ALGORITHM:
Perceptual Hash (pHash) using DCT:
1. Resize frame to 32×32 grayscale
2. Apply 2D DCT
3. Take top-left 8×8 DCT coefficients (low frequency)
4. Binarize above/below the mean of those coefficients
5. Hash = 64-bit integer

Hamming distance between two hashes:
  - 0: identical
  - ≤ 8: very similar (near-duplicate)
  - 9-20: similar but noticeably different
  - > 20: different frames

Reference:
  Zauner, C. (2010). Implementation and benchmarking of perceptual image hash
  functions. Upper Austria University of Applied Sciences Technical Report.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def perceptual_hash(frame_rgb: NDArray[np.uint8]) -> int:
    """
    Compute perceptual (DCT-based) hash for a frame.

    Args:
        frame_rgb: RGB image, uint8.

    Returns:
        64-bit integer perceptual hash.
    """
    # Step 1: resize to 32×32
    small = cv2.resize(frame_rgb, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Step 2: DCT
    dct = cv2.dct(gray)

    # Step 3: top-left 8×8
    dct_low = dct[:8, :8]

    # Step 4: compute mean excluding DC component (top-left element)
    mean_val = (dct_low.sum() - dct_low[0, 0]) / 63.0

    # Step 5: binarize and pack into integer
    bits = (dct_low > mean_val).flatten()
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)

    return int(hash_val)


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Compute Hamming distance between two 64-bit perceptual hashes.

    Args:
        hash1, hash2: 64-bit integer hashes.

    Returns:
        Number of differing bits (0–64).
    """
    xor = hash1 ^ hash2
    # Count set bits (popcount)
    count = 0
    while xor:
        xor &= xor - 1
        count += 1
    return count


def is_near_duplicate(
    hash1: int,
    hash2: int,
    threshold: int = 8,
) -> bool:
    """
    Return True if two frames are near-duplicates.

    Args:
        hash1, hash2: Perceptual hashes.
        threshold:    Maximum Hamming distance for "near-duplicate".
                      Default 8 corresponds to ~12.5% bit difference.
    """
    return hamming_distance(hash1, hash2) <= threshold


def deduplicate_frames(
    hashes: List[int],
    *,
    threshold: int = 8,
    min_gap: int = 1,
) -> List[int]:
    """
    Return indices of non-duplicate frames.

    Greedy deduplication: keep the first frame in any near-duplicate cluster.

    Args:
        hashes:    List of perceptual hashes in frame order.
        threshold: Hamming distance threshold for "duplicate".
        min_gap:   Minimum frame gap between selected frames (prevents
                   selecting two frames that are adjacent even if different).

    Returns:
        List of indices into `hashes` that are not duplicates.
    """
    if not hashes:
        return []

    selected: List[int] = [0]
    last_hash = hashes[0]
    last_idx = 0

    for i, h in enumerate(hashes[1:], start=1):
        if (i - last_idx) < min_gap:
            continue
        if not is_near_duplicate(h, last_hash, threshold):
            selected.append(i)
            last_hash = h
            last_idx = i

    return selected


def find_scene_changes(
    hashes: List[int],
    *,
    threshold: int = 25,
) -> List[int]:
    """
    Detect scene changes from a sequence of perceptual hashes.

    A scene change is defined as a frame where the Hamming distance
    from the previous frame exceeds the threshold.

    Args:
        hashes:    List of perceptual hashes in frame order.
        threshold: Hamming distance threshold for scene change detection.

    Returns:
        List of frame indices where scene changes occur.
    """
    if len(hashes) < 2:
        return []

    scene_changes: List[int] = []
    for i in range(1, len(hashes)):
        if hamming_distance(hashes[i - 1], hashes[i]) > threshold:
            scene_changes.append(i)

    return scene_changes
