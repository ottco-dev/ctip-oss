"""
measurement.domain.profile_manager — Microscope calibration profile management.

A "profile" maps a specific microscope / objective / camera combination to a
calibrated pixels-per-micrometer (px/µm) value.  Multiple profiles can be
stored and selected per imaging session.

CALIBRATION METHODS SUPPORTED:
1. Stage micrometer — gold standard: image a calibrated graticule and measure
   the known scale bar.  Supported via measurement.calibration.stage_micrometer.
2. Known reference object — measure an object of known physical size.
3. Manual entry — directly specify µm-per-pixel from manufacturer data-sheet.

FILE FORMAT:
Profiles are stored as JSON:
  {
    "profiles": {
      "profile_id": {
        "name": "Olympus BX53 40x",
        "um_per_pixel": 0.1625,
        "objective": "40x",
        "camera": "Axiocam 506",
        "image_width": 2448,
        "image_height": 2048,
        "calibration_method": "stage_micrometer",
        "calibration_date": "2026-05-01",
        "notes": "Stage micr. 100µm/div at 40x",
        "uncertainty_um": 0.005
      }
    },
    "default_profile": "profile_id"
  }

Reference:
  Abramowitz, M. & Davidson, M.W. (2012). Microscopy Resource Center:
  Calibration and Measurement. Olympus America Inc.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MicroscopeProfile:
    """
    A calibrated microscope configuration profile.

    Stores the pixel→µm scaling factor and all metadata needed to
    reproduce measurements across imaging sessions.
    """

    profile_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Default Profile"

    um_per_pixel: float = 1.0
    """
    Micrometers per pixel (µm/px).
    Typical values:
      10× objective:  ~0.65 µm/px
      20× objective:  ~0.32 µm/px
      40× objective:  ~0.16 µm/px
      100× objective: ~0.065 µm/px
    """

    objective: str = ""
    """Objective magnification string, e.g. '40x', '100x oil'."""

    camera: str = ""
    """Camera model/sensor identifier."""

    image_width: Optional[int] = None
    """Expected image width in pixels (for validation)."""

    image_height: Optional[int] = None
    """Expected image height in pixels (for validation)."""

    calibration_method: str = "manual"
    """One of: 'stage_micrometer', 'reference_object', 'manual'."""

    calibration_date: Optional[str] = None
    """ISO 8601 date string of last calibration."""

    uncertainty_um: Optional[float] = None
    """
    Estimated calibration uncertainty in µm/px.
    Derived from stage micrometer measurement error or instrument specs.
    """

    notes: str = ""

    @property
    def px_per_um(self) -> float:
        """Inverse of um_per_pixel. Pixels per micrometer."""
        if self.um_per_pixel <= 0:
            raise ValueError("um_per_pixel must be positive")
        return 1.0 / self.um_per_pixel

    def px_to_um(self, pixels: float) -> float:
        """Convert pixel distance to micrometers."""
        return pixels * self.um_per_pixel

    def um_to_px(self, microns: float) -> float:
        """Convert micrometers to pixel distance."""
        return microns / self.um_per_pixel if self.um_per_pixel > 0 else 0.0

    def area_px_to_um2(self, area_px: float) -> float:
        """Convert pixel area to µm²."""
        return area_px * (self.um_per_pixel ** 2)

    def area_um2_to_px(self, area_um2: float) -> float:
        """Convert µm² to pixel area."""
        return area_um2 / (self.um_per_pixel ** 2) if self.um_per_pixel > 0 else 0.0

    def validate_image_size(self, width: int, height: int) -> bool:
        """
        Check if an image matches the expected dimensions for this profile.

        Returns True if sizes match or no expected size is configured.
        """
        if self.image_width is None or self.image_height is None:
            return True
        return abs(width - self.image_width) <= 4 and abs(height - self.image_height) <= 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MicroscopeProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Built-in default profiles for common microscope setups
DEFAULT_PROFILES: Dict[str, MicroscopeProfile] = {
    "10x_generic": MicroscopeProfile(
        profile_id="10x_generic",
        name="Generic 10× objective",
        um_per_pixel=0.65,
        objective="10x",
        calibration_method="manual",
        notes="Generic estimate for standard 10× objective",
        uncertainty_um=0.05,
    ),
    "20x_generic": MicroscopeProfile(
        profile_id="20x_generic",
        name="Generic 20× objective",
        um_per_pixel=0.325,
        objective="20x",
        calibration_method="manual",
        notes="Generic estimate for standard 20× objective",
        uncertainty_um=0.025,
    ),
    "40x_generic": MicroscopeProfile(
        profile_id="40x_generic",
        name="Generic 40× objective",
        um_per_pixel=0.1625,
        objective="40x",
        calibration_method="manual",
        notes="Generic estimate for standard 40× objective",
        uncertainty_um=0.01,
    ),
    "100x_generic": MicroscopeProfile(
        profile_id="100x_generic",
        name="Generic 100× oil immersion",
        um_per_pixel=0.065,
        objective="100x oil",
        calibration_method="manual",
        notes="Generic estimate for 100× oil objective",
        uncertainty_um=0.005,
    ),
}


class ProfileManager:
    """
    Manages a collection of microscope calibration profiles.

    Profiles are persisted to a JSON file and loaded on demand.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path
        self._profiles: Dict[str, MicroscopeProfile] = dict(DEFAULT_PROFILES)
        self._default_id: Optional[str] = "40x_generic"

        if storage_path and Path(storage_path).exists():
            self._load(storage_path)

    # ── CRUD ─────────────────────────────────────────────────────────

    def add_profile(self, profile: MicroscopeProfile, *, set_default: bool = False) -> str:
        """Add or update a profile. Returns profile_id."""
        self._profiles[profile.profile_id] = profile
        if set_default:
            self._default_id = profile.profile_id
        self._save()
        return profile.profile_id

    def get_profile(self, profile_id: str) -> Optional[MicroscopeProfile]:
        """Retrieve a profile by ID. Returns None if not found."""
        return self._profiles.get(profile_id)

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile. Returns True if it existed."""
        if profile_id in DEFAULT_PROFILES:
            raise ValueError(f"Cannot delete built-in profile '{profile_id}'")
        existed = profile_id in self._profiles
        self._profiles.pop(profile_id, None)
        if self._default_id == profile_id:
            self._default_id = "40x_generic"
        self._save()
        return existed

    def list_profiles(self) -> list[MicroscopeProfile]:
        """Return all profiles sorted by name."""
        return sorted(self._profiles.values(), key=lambda p: p.name)

    @property
    def default_profile(self) -> Optional[MicroscopeProfile]:
        """Return the default profile."""
        if self._default_id:
            return self._profiles.get(self._default_id)
        return None

    def set_default(self, profile_id: str) -> None:
        if profile_id not in self._profiles:
            raise KeyError(f"Profile '{profile_id}' not found")
        self._default_id = profile_id
        self._save()

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._path:
            return
        path = Path(self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profiles": {
                pid: p.to_dict()
                for pid, p in self._profiles.items()
                if pid not in DEFAULT_PROFILES  # Don't persist built-ins
            },
            "default_profile": self._default_id,
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(Path(path).read_text())
            for pid, pdata in data.get("profiles", {}).items():
                self._profiles[pid] = MicroscopeProfile.from_dict(pdata)
            self._default_id = data.get("default_profile", self._default_id)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Profile load failed: {e}")

    def create_from_stage_micrometer(
        self,
        name: str,
        scale_bar_px: float,
        scale_bar_um: float,
        *,
        objective: str = "",
        camera: str = "",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        notes: str = "",
        set_default: bool = False,
    ) -> MicroscopeProfile:
        """
        Create a calibrated profile from a stage micrometer measurement.

        Args:
            name:           Human-readable profile name.
            scale_bar_px:   Measured scale bar length in pixels.
            scale_bar_um:   Known physical length of the scale bar in µm.
            objective:      Objective magnification string.
            camera:         Camera model string.
            image_width:    Expected image width in pixels.
            image_height:   Expected image height in pixels.
            notes:          Free-text notes about the calibration.
            set_default:    If True, set this profile as the default.

        Returns:
            Newly created MicroscopeProfile.
        """
        if scale_bar_px <= 0:
            raise ValueError("scale_bar_px must be positive")
        if scale_bar_um <= 0:
            raise ValueError("scale_bar_um must be positive")

        um_per_pixel = scale_bar_um / scale_bar_px

        # Uncertainty: 1-pixel measurement error propagated
        uncertainty = um_per_pixel / scale_bar_px  # ≈ 1px/scale_bar_px × µm/px

        profile = MicroscopeProfile(
            name=name,
            um_per_pixel=um_per_pixel,
            objective=objective,
            camera=camera,
            image_width=image_width,
            image_height=image_height,
            calibration_method="stage_micrometer",
            calibration_date=str(date.today()),
            uncertainty_um=uncertainty,
            notes=notes,
        )

        self.add_profile(profile, set_default=set_default)
        return profile
