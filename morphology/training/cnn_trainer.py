"""
morphology.training.cnn_trainer — CNN fine-tuning pipeline for trichome morphology.

Fine-tunes EfficientNet-B0 (torchvision) or MobileNetV3-Small for classifying
trichome crops into four morphological categories.

Target hardware:  RTX 4060 8 GB / i5-13400F / 16 GB RAM
Mixed precision:  torch.cuda.amp (FP16) — halves VRAM usage
Reproducibility:  GLOBAL_SEED=42

Directory layout expected at data_dir:
    data/morphology_crops/
        capitate_stalked/   *.jpg *.png
        capitate_sessile/
        bulbous/
        non_glandular/
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Class mapping ────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "capitate_stalked",
    "capitate_sessile",
    "bulbous",
    "non_glandular",
]

CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class MorphologyCNNConfig:
    """
    Full configuration for CNN morphology classifier training.

    All parameters have scientifically-motivated defaults for microscopy crop
    classification on RTX 4060 hardware.
    """

    # ── Architecture ──────────────────────────────────────────────────
    model_arch: str = "efficientnet_b0"
    """Backbone architecture. Options: 'efficientnet_b0', 'mobilenet_v3_small'."""

    num_classes: int = 4
    """Output classes: CAPITATE_STALKED, CAPITATE_SESSILE, BULBOUS, NON_GLANDULAR."""

    input_size: int = 224
    """Input spatial resolution in pixels (square crop)."""

    dropout: float = 0.3
    """Dropout probability before the classification head."""

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 32
    """Mini-batch size. Reduce to 16 if OOM on RTX 4060 with large input."""

    learning_rate: float = 1e-4
    """AdamW initial learning rate."""

    epochs: int = 50
    """Maximum training epochs."""

    early_stopping_patience: int = 10
    """Stop training if validation loss does not improve for this many epochs."""

    val_split: float = 0.2
    """Fraction of dataset to reserve for validation."""

    seed: int = 42
    """Global random seed for reproducibility."""

    # ── Hardware ──────────────────────────────────────────────────────
    use_fp16: bool = True
    """Enable FP16 mixed-precision training (cuts VRAM usage ~50%)."""

    num_workers: int = 4
    """DataLoader worker count for parallel image loading."""

    # ── Augmentation ──────────────────────────────────────────────────
    augment: bool = True
    """Apply microscopy-specific augmentation pipeline during training."""

    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: str = "./data/morphology_crops"
    """Root directory with one subdirectory per class."""

    output_dir: str = "./data/models/morphology"
    """Directory to save checkpoints, training history, and ONNX exports."""

    def __post_init__(self) -> None:
        # Validate arch
        valid_archs = {"efficientnet_b0", "mobilenet_v3_small"}
        if self.model_arch not in valid_archs:
            raise ValueError(
                f"model_arch must be one of {valid_archs}, got '{self.model_arch}'"
            )
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


# ── Trainer ───────────────────────────────────────────────────────────────────


class MorphologyCNNTrainer:
    """
    Fine-tunes EfficientNet-B0 or MobileNetV3-Small for trichome morphology
    classification.

    Usage:
        config = MorphologyCNNConfig(data_dir="./data/morphology_crops")
        trainer = MorphologyCNNTrainer(config)
        loaders = trainer.prepare_data()
        model   = trainer.build_model()
        history = trainer.train()
        metrics = trainer.evaluate("./data/models/morphology/best.pt")
        onnx_path = trainer.export_onnx("best.pt", "morphology.onnx")
    """

    def __init__(self, config: MorphologyCNNConfig) -> None:
        self.config = config
        self._training_history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self._status: dict[str, Any] = {
            "state": "idle",
            "epoch": 0,
            "best_val_loss": float("inf"),
            "early_stop_counter": 0,
        }
        self._set_seed(config.seed)

    # ── Seed ──────────────────────────────────────────────────────────

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set all relevant random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    # ── Data preparation ──────────────────────────────────────────────

    def prepare_data(self) -> tuple[Any, Any]:
        """
        Build train and validation DataLoaders from the directory structure.

        Expects:
            {data_dir}/capitate_stalked/*.{jpg,png}
            {data_dir}/capitate_sessile/*.{jpg,png}
            {data_dir}/bulbous/*.{jpg,png}
            {data_dir}/non_glandular/*.{jpg,png}

        Returns:
            (train_loader, val_loader) — DataLoader objects.
        """
        import torch
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, transforms

        train_tf = self._build_train_transforms()
        val_tf = self._build_val_transforms()

        data_path = Path(self.config.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"data_dir does not exist: {data_path}. "
                "Create class subdirectories with crop images before training."
            )

        # Load full dataset with train transforms; we'll swap val subset transforms below
        full_dataset = datasets.ImageFolder(
            root=str(data_path),
            transform=train_tf,
        )

        if len(full_dataset) == 0:
            raise ValueError(f"No images found under {data_path}")

        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self.config.val_split))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self.config.seed)
        train_subset, val_subset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

        # Apply separate val transforms via a lightweight wrapper
        val_subset = _SubsetWithTransform(val_subset, val_tf)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=len(train_subset) >= self.config.batch_size,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            "Dataset prepared: %d train / %d val images across %d classes",
            n_train,
            n_val,
            len(full_dataset.classes),
        )
        return train_loader, val_loader

    def _build_train_transforms(self) -> Any:
        """Microscopy-specific augmentation pipeline for training."""
        from torchvision import transforms

        tf_list = [
            transforms.Resize((self.config.input_size, self.config.input_size)),
        ]

        if self.config.augment:
            tf_list += [
                # Microscope has no canonical orientation
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
                # Illumination variation (brightfield / darkfield / fluorescence)
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.0
                ),
                # Phase-contrast images are greyscale; simulate with low probability
                transforms.RandomGrayscale(p=0.1),
            ]

        tf_list += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(tf_list)

    def _build_val_transforms(self) -> Any:
        """Deterministic validation transforms (no augmentation)."""
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize((self.config.input_size, self.config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ── Model construction ────────────────────────────────────────────

    def build_model(self) -> Any:
        """
        Load a pretrained backbone from torchvision and replace the classifier
        head to output `num_classes` logits.

        Tries timm first (richer EfficientNet-Lite support), falls back to
        torchvision. Both EfficientNet-B0 and MobileNetV3-Small are available
        in torchvision and have been validated on this task.

        Returns:
            nn.Module — model ready for fine-tuning.
        """
        import torch.nn as nn

        arch = self.config.model_arch
        n_cls = self.config.num_classes
        drop = self.config.dropout

        # Attempt timm first for broader architecture support
        try:
            import timm

            timm_name_map = {
                "efficientnet_b0": "efficientnet_b0",
                "mobilenet_v3_small": "mobilenetv3_small_100",
            }
            model = timm.create_model(
                timm_name_map[arch],
                pretrained=True,
                num_classes=n_cls,      # timm handles head replacement correctly
                drop_rate=drop,
            )
            logger.info("Built %s via timm (→%d classes)", arch, n_cls)
            return model

        except ImportError:
            pass  # Fall back to torchvision

        # torchvision fallback
        from torchvision import models

        if arch == "efficientnet_b0":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=drop, inplace=True),
                nn.Linear(in_features, n_cls),
            )

        elif arch == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, n_cls)
            # Insert dropout before final linear
            model.classifier = nn.Sequential(
                model.classifier[0],
                model.classifier[1],
                model.classifier[2],
                nn.Dropout(p=drop, inplace=False),
                nn.Linear(in_features, n_cls),
            )

        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        logger.info("Built %s via torchvision (→%d classes)", arch, n_cls)
        return model

    # ── Training loop ─────────────────────────────────────────────────

    def train(
        self,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            progress_callback: Optional callable called after each epoch with
                a status dict containing epoch, train_loss, val_loss, etc.

        Returns:
            Training history dict with per-epoch metrics and final summary.
        """
        import torch
        import torch.nn as nn
        from torch.cuda.amp import GradScaler, autocast

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        use_amp = self.config.use_fp16 and device.type == "cuda"

        self._status["state"] = "preparing_data"
        train_loader, val_loader = self.prepare_data()

        model = self.build_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=1e-6
        )
        scaler = GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_ckpt_path = output_dir / "best.pt"
        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        self._status["state"] = "training"
        logger.info(
            "Training %s on %s | FP16=%s | epochs=%d",
            self.config.model_arch,
            device,
            use_amp,
            self.config.epochs,
        )

        for epoch in range(1, self.config.epochs + 1):
            # ── Train ──────────────────────────────────────────────────
            model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                t_loss += loss.item() * imgs.size(0)
                t_correct += (logits.argmax(1) == labels).sum().item()
                t_total += imgs.size(0)

            train_loss = t_loss / max(t_total, 1)
            train_acc = t_correct / max(t_total, 1)

            # ── Validate ───────────────────────────────────────────────
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    with autocast(enabled=use_amp):
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                    v_loss += loss.item() * imgs.size(0)
                    v_correct += (logits.argmax(1) == labels).sum().item()
                    v_total += imgs.size(0)

            val_loss = v_loss / max(v_total, 1)
            val_acc = v_correct / max(v_total, 1)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            self._training_history = history
            self._status.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                best_val_loss=best_val_loss,
                early_stop_counter=patience_counter,
            )

            logger.info(
                "Epoch %3d/%d — train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f",
                epoch,
                self.config.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if progress_callback:
                try:
                    progress_callback(dict(self._status))
                except Exception:
                    pass

            # ── Checkpoint / early stopping ────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "config": self.config.__dict__,
                    },
                    best_ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch,
                        self.config.early_stopping_patience,
                    )
                    break

        # Save training history to JSON for reproducibility
        history_path = output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": history["val_acc"][best_epoch - 1] if best_epoch > 0 else 0.0,
            "final_epoch": epoch,
            "checkpoint_path": str(best_ckpt_path),
            "history_path": str(history_path),
            "history": history,
        }
        self._status["state"] = "completed"
        self._status["summary"] = summary
        return summary

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, model_path: str) -> dict[str, Any]:
        """
        Evaluate a saved checkpoint on the validation set.

        Args:
            model_path: Path to a .pt checkpoint saved by train().

        Returns:
            Dict with per-class accuracy, confusion matrix, top-1/top-5 accuracy,
            and per-class precision/recall/F1.
        """
        import torch
        from torchvision import transforms

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = self.build_model().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Prepare validation data
        _, val_loader = self.prepare_data()

        all_preds: list[int] = []
        all_labels: list[int] = []
        all_logits: list[Any] = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = logits.argmax(1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_logits.extend(logits.cpu().tolist())

        import numpy as np

        n_cls = self.config.num_classes
        labels_arr = np.array(all_labels)
        preds_arr = np.array(all_preds)

        # Confusion matrix
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for true, pred in zip(all_labels, all_preds):
            cm[true][pred] += 1

        # Per-class metrics
        per_class: dict[str, dict] = {}
        for c_idx in range(n_cls):
            tp = int(cm[c_idx, c_idx])
            fp = int(cm[:, c_idx].sum() - tp)
            fn = int(cm[c_idx, :].sum() - tp)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            cls_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else str(c_idx)
            per_class[cls_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int((labels_arr == c_idx).sum()),
            }

        # Top-1 accuracy
        top1_acc = float((preds_arr == labels_arr).mean())

        # Top-5 (only meaningful if num_classes >= 5)
        top5_acc: float | None = None
        if n_cls >= 5 and all_logits:
            logits_arr = np.array(all_logits)
            top5_preds = np.argsort(logits_arr, axis=1)[:, -5:]
            top5_acc = float(
                np.array(
                    [lbl in top5_preds[i] for i, lbl in enumerate(all_labels)]
                ).mean()
            )

        return {
            "top1_accuracy": round(top1_acc, 4),
            "top5_accuracy": top5_acc,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "num_samples": len(all_labels),
        }

    # ── ONNX export ───────────────────────────────────────────────────

    def export_onnx(self, model_path: str, output_path: str) -> str:
        """
        Export a trained checkpoint to ONNX format for production inference.

        Args:
            model_path: Path to .pt checkpoint.
            output_path: Destination .onnx file path.

        Returns:
            Absolute path to the exported .onnx file.
        """
        import torch

        device = torch.device("cpu")  # export on CPU for portability

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = self.build_model().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(
            1, 3, self.config.input_size, self.config.input_size,
            device=device,
        )

        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(output_p),
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        logger.info("ONNX model exported to %s", output_p)
        return str(output_p.resolve())

    # ── Status accessors ──────────────────────────────────────────────

    @property
    def status(self) -> dict[str, Any]:
        """Return a snapshot of the current training status."""
        return dict(self._status)

    @property
    def training_history(self) -> dict[str, list]:
        """Return recorded per-epoch training history."""
        return dict(self._training_history)


# ── Dataset utility ───────────────────────────────────────────────────────────


class _SubsetWithTransform:
    """
    Wraps a torch.utils.data.Subset and overrides transforms.

    Required because torchvision ImageFolder applies transforms at dataset
    level, not subset level — we need separate augmentation for train / val.
    """

    def __init__(self, subset: Any, transform: Any) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img, label = self.subset[idx]
        # img is already a Tensor from ImageFolder's transform;
        # re-apply original PIL loading by going through dataset directly.
        # We access the underlying dataset and its loader.
        dataset = self.subset.dataset
        actual_idx = self.subset.indices[idx]
        path, label = dataset.samples[actual_idx]

        from PIL import Image as PILImage
        img_pil = PILImage.open(path).convert("RGB")
        img_t = self.transform(img_pil)
        return img_t, label
