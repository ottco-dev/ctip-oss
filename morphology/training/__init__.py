"""
morphology.training — CNN training pipeline for trichome morphology classification.

Provides fine-tuning of EfficientNet-B0 / MobileNetV3-Small backbones for
classifying trichome crops into four morphological classes:
  - CAPITATE_STALKED
  - CAPITATE_SESSILE
  - BULBOUS
  - NON_GLANDULAR

Design goals:
  - RTX 4060 (8 GB VRAM) compatible via FP16 mixed precision
  - Microscopy-specific augmentation pipeline
  - Early stopping with configurable patience
  - ONNX export for production inference
  - Full reproducibility via fixed seed
"""

from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer

__all__ = ["MorphologyCNNConfig", "MorphologyCNNTrainer"]
