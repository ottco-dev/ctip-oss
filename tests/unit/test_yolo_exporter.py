"""
tests/unit/test_yolo_exporter.py — Unit tests for inference.tensorrt_engine.exporter.

Tests cover:
  - YOLOExportConfig validation (valid + invalid params)
  - YOLOToTensorRT.export_onnx_only: happy path, missing model, missing ultralytics
  - YOLOToTensorRT.export: full pipeline, TRT absent graceful degradation
  - YOLOToTensorRT.validate_onnx: pass, fail, missing onnxruntime, missing file
  - _simplify_onnx: success, onnxsim absent, simplification failure
  - CLI commands via CliRunner: onnx, tensorrt, validate
  - Edge cases: dynamic_batch, opset limits, workspace_gb limits

All ultralytics.YOLO and tensorrt calls are mocked.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from typer.testing import CliRunner

# Module under test
from inference.tensorrt_engine.exporter import (
    YOLOExportConfig,
    YOLOToTensorRT,
    ExportError,
)
from apps.cli.commands.convert import app as convert_app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_model(tmp_path) -> Path:
    """Create a fake .pt model file."""
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"fake pt weights")
    return pt


@pytest.fixture()
def tmp_onnx(tmp_path) -> Path:
    """Create a fake .onnx file."""
    onnx_file = tmp_path / "best.onnx"
    onnx_file.write_bytes(b"fake onnx model")
    return onnx_file


@pytest.fixture()
def valid_config(tmp_model, tmp_path) -> YOLOExportConfig:
    return YOLOExportConfig(
        model_path=str(tmp_model),
        output_dir=str(tmp_path / "out"),
    )


# ---------------------------------------------------------------------------
# YOLOExportConfig validation
# ---------------------------------------------------------------------------

class TestYOLOExportConfig:
    def test_valid_defaults(self, tmp_model, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(tmp_path),
        )
        assert cfg.imgsz == 1280
        assert cfg.fp16 is True
        assert cfg.opset == 17
        assert cfg.simplify is True
        assert cfg.workspace_gb == 4.0
        assert cfg.dynamic_batch is False

    def test_custom_params(self, tmp_model, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(tmp_path),
            imgsz=640,
            fp16=False,
            opset=12,
            simplify=False,
            workspace_gb=6.0,
            dynamic_batch=True,
        )
        assert cfg.imgsz == 640
        assert cfg.fp16 is False
        assert cfg.opset == 12
        assert cfg.dynamic_batch is True

    def test_empty_model_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="model_path"):
            YOLOExportConfig(model_path="", output_dir=str(tmp_path))

    def test_empty_output_dir_raises(self, tmp_model):
        with pytest.raises(ValueError, match="output_dir"):
            YOLOExportConfig(model_path=str(tmp_model), output_dir="")

    def test_zero_imgsz_raises(self, tmp_model, tmp_path):
        with pytest.raises(ValueError, match="imgsz"):
            YOLOExportConfig(model_path=str(tmp_model), output_dir=str(tmp_path), imgsz=0)

    def test_negative_imgsz_raises(self, tmp_model, tmp_path):
        with pytest.raises(ValueError, match="imgsz"):
            YOLOExportConfig(model_path=str(tmp_model), output_dir=str(tmp_path), imgsz=-1)

    def test_invalid_opset_low_raises(self, tmp_model, tmp_path):
        with pytest.raises(ValueError, match="opset"):
            YOLOExportConfig(model_path=str(tmp_model), output_dir=str(tmp_path), opset=0)

    def test_invalid_opset_high_raises(self, tmp_model, tmp_path):
        with pytest.raises(ValueError, match="opset"):
            YOLOExportConfig(model_path=str(tmp_model), output_dir=str(tmp_path), opset=25)

    def test_zero_workspace_gb_raises(self, tmp_model, tmp_path):
        with pytest.raises(ValueError, match="workspace_gb"):
            YOLOExportConfig(model_path=str(tmp_model), output_dir=str(tmp_path), workspace_gb=0)


# ---------------------------------------------------------------------------
# export_onnx_only
# ---------------------------------------------------------------------------

class TestExportOnnxOnly:
    def test_missing_model_raises_file_not_found(self, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_path / "nonexistent.pt"),
            output_dir=str(tmp_path / "out"),
        )
        exporter = YOLOToTensorRT(cfg)
        with pytest.raises(FileNotFoundError, match="nonexistent.pt"):
            exporter.export_onnx_only()

    def test_missing_ultralytics_raises_export_error(self, tmp_model, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(tmp_path / "out"),
        )
        exporter = YOLOToTensorRT(cfg)
        # Simulate ultralytics not installed by setting module-level YOLO to None
        with patch("inference.tensorrt_engine.exporter.YOLO", None):
            with pytest.raises(ExportError, match="ultralytics"):
                exporter.export_onnx_only()

    def test_successful_onnx_export(self, tmp_model, tmp_path):
        """Mock YOLO.export to return an onnx file already in output_dir."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        onnx_out = out_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx")

        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(out_dir),
            simplify=False,
        )
        exporter = YOLOToTensorRT(cfg)

        mock_yolo = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(onnx_out)
        mock_yolo.return_value = mock_yolo_instance

        with patch("inference.tensorrt_engine.exporter.YOLO", mock_yolo):
            result = exporter.export_onnx_only()

        assert result == str(onnx_out)

    def test_onnx_export_creates_output_dir(self, tmp_model, tmp_path):
        """Output directory should be created automatically."""
        out_dir = tmp_path / "nested" / "output"
        onnx_out = out_dir / "best.onnx"

        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(out_dir),
            simplify=False,
        )
        exporter = YOLOToTensorRT(cfg)

        mock_yolo_instance = MagicMock()
        # Simulate ultralytics placing the file next to the .pt:
        onnx_sibling = tmp_model.parent / "best.onnx"
        onnx_sibling.write_bytes(b"fake onnx")
        mock_yolo_instance.export.return_value = str(onnx_sibling)

        with patch("inference.tensorrt_engine.exporter.YOLO", return_value=mock_yolo_instance):
            result = exporter.export_onnx_only()

        assert out_dir.exists()
        assert Path(result).parent == out_dir

    def test_export_calls_yolo_with_correct_params(self, tmp_model, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        onnx_out = out_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx")

        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(out_dir),
            imgsz=640,
            opset=12,
            fp16=False,
            simplify=False,
            dynamic_batch=True,
        )
        exporter = YOLOToTensorRT(cfg)
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(onnx_out)

        with patch("inference.tensorrt_engine.exporter.YOLO", return_value=mock_yolo_instance):
            exporter.export_onnx_only()

        call_kwargs = mock_yolo_instance.export.call_args[1]
        assert call_kwargs["format"] == "onnx"
        assert call_kwargs["imgsz"] == 640
        assert call_kwargs["opset"] == 12
        assert call_kwargs["half"] is False
        assert call_kwargs["dynamic"] is True


# ---------------------------------------------------------------------------
# export (full pipeline)
# ---------------------------------------------------------------------------

class TestFullExport:
    def test_export_without_tensorrt_returns_onnx_only(self, tmp_model, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        onnx_out = out_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx")

        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(out_dir),
            simplify=False,
        )
        exporter = YOLOToTensorRT(cfg)
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(onnx_out)

        # Simulate TRT absent by making the `import tensorrt` line raise ImportError.
        # We do this by temporarily removing tensorrt from sys.modules and replacing
        # it with a sentinel that raises on import.
        import importlib
        real_trt = sys.modules.get("tensorrt")
        sys.modules["tensorrt"] = None  # type: ignore[assignment]
        try:
            with patch("inference.tensorrt_engine.exporter.YOLO", return_value=mock_yolo_instance):
                result = exporter.export()
        finally:
            if real_trt is None:
                sys.modules.pop("tensorrt", None)
            else:
                sys.modules["tensorrt"] = real_trt

        assert result["onnx_path"] == str(onnx_out)
        assert result["engine_path"] is None
        assert result["export_time_s"] >= 0

    def test_export_result_keys(self, tmp_model, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        onnx_out = out_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx")

        cfg = YOLOExportConfig(
            model_path=str(tmp_model),
            output_dir=str(out_dir),
            simplify=False,
        )
        exporter = YOLOToTensorRT(cfg)
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(onnx_out)

        import importlib
        real_trt = sys.modules.get("tensorrt")
        sys.modules["tensorrt"] = None  # type: ignore[assignment]
        try:
            with patch("inference.tensorrt_engine.exporter.YOLO", return_value=mock_yolo_instance):
                result = exporter.export()
        finally:
            if real_trt is None:
                sys.modules.pop("tensorrt", None)
            else:
                sys.modules["tensorrt"] = real_trt

        assert "onnx_path" in result
        assert "engine_path" in result
        assert "export_time_s" in result

    def test_export_propagates_file_not_found(self, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_path / "missing.pt"),
            output_dir=str(tmp_path / "out"),
        )
        exporter = YOLOToTensorRT(cfg)
        with pytest.raises(FileNotFoundError):
            exporter.export()


# ---------------------------------------------------------------------------
# validate_onnx
# ---------------------------------------------------------------------------

class TestValidateOnnx:
    def test_valid_onnx_returns_ok_true(self, tmp_onnx, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_onnx),
            output_dir=str(tmp_path),
        )
        exporter = YOLOToTensorRT(cfg)

        # Mock onnxruntime session
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_input.shape = [1, 3, 640, 640]

        mock_output = np.zeros((1, 8, 8400), dtype=np.float32)
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [mock_output]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        # Patch module-level `ort` reference directly (no sys.modules manipulation needed)
        with patch("inference.tensorrt_engine.exporter.ort", mock_ort):
            result = exporter.validate_onnx(str(tmp_onnx))

        assert result["ok"] is True
        assert isinstance(result["output_shapes"], list)
        assert result["error"] is None

    def test_missing_file_returns_ok_false(self, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_path / "dummy.pt"),
            output_dir=str(tmp_path),
        )
        # Create dummy .pt so config doesn't raise in __post_init__
        (tmp_path / "dummy.pt").write_bytes(b"fake")
        exporter = YOLOToTensorRT(cfg)

        result = exporter.validate_onnx(str(tmp_path / "nonexistent.onnx"))
        assert result["ok"] is False
        assert result["error"] is not None

    def test_missing_onnxruntime_returns_ok_false(self, tmp_onnx, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_onnx),
            output_dir=str(tmp_path),
        )
        exporter = YOLOToTensorRT(cfg)

        # Simulate onnxruntime not installed by setting module-level ort to None
        with patch("inference.tensorrt_engine.exporter.ort", None):
            result = exporter.validate_onnx(str(tmp_onnx))

        assert result["ok"] is False
        assert result["error"] is not None

    def test_runtime_error_returns_ok_false(self, tmp_onnx, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_onnx),
            output_dir=str(tmp_path),
        )
        exporter = YOLOToTensorRT(cfg)

        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("invalid onnx model")

        with patch("inference.tensorrt_engine.exporter.ort", mock_ort):
            result = exporter.validate_onnx(str(tmp_onnx))

        assert result["ok"] is False
        assert "invalid onnx model" in result["error"]


# ---------------------------------------------------------------------------
# _simplify_onnx
# ---------------------------------------------------------------------------

class TestSimplifyOnnx:
    def test_simplify_skipped_when_onnxsim_absent(self, tmp_onnx, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_onnx),
            output_dir=str(tmp_path),
        )
        exporter = YOLOToTensorRT(cfg)

        # Simulate onnxsim absent by setting module-level onnxsim to None
        with patch("inference.tensorrt_engine.exporter.onnxsim", None):
            result = exporter._simplify_onnx(str(tmp_onnx))

        # Should return original path unchanged when onnxsim is absent
        assert result == str(tmp_onnx)

    def test_simplify_saves_model_when_check_ok(self, tmp_onnx, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_onnx),
            output_dir=str(tmp_path),
        )
        exporter = YOLOToTensorRT(cfg)

        mock_model = MagicMock()
        mock_simplified = MagicMock()

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model

        mock_onnxsim = MagicMock()
        mock_onnxsim.simplify.return_value = (mock_simplified, True)

        with (
            patch("inference.tensorrt_engine.exporter.onnx", mock_onnx),
            patch("inference.tensorrt_engine.exporter.onnxsim", mock_onnxsim),
        ):
            result = exporter._simplify_onnx(str(tmp_onnx))

        assert result == str(tmp_onnx)

    def test_simplify_exception_handled_gracefully(self, tmp_onnx, tmp_path):
        cfg = YOLOExportConfig(
            model_path=str(tmp_onnx),
            output_dir=str(tmp_path),
        )
        exporter = YOLOToTensorRT(cfg)

        mock_onnx = MagicMock()
        mock_onnx.load.side_effect = RuntimeError("corrupt model")

        mock_onnxsim = MagicMock()

        with (
            patch("inference.tensorrt_engine.exporter.onnx", mock_onnx),
            patch("inference.tensorrt_engine.exporter.onnxsim", mock_onnxsim),
        ):
            # Should not raise; returns original path
            result = exporter._simplify_onnx(str(tmp_onnx))

        assert result == str(tmp_onnx)


# ---------------------------------------------------------------------------
# CLI commands via CliRunner
# ---------------------------------------------------------------------------

class TestConvertCLI:
    def test_onnx_command_missing_model_exits_1(self, tmp_path):
        result = runner.invoke(convert_app, [
            "onnx", str(tmp_path / "missing.pt"),
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_onnx_command_success(self, tmp_model, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        onnx_out = out_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx")

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(onnx_out)

        with patch("inference.tensorrt_engine.exporter.YOLO", return_value=mock_yolo_instance):
            result = runner.invoke(convert_app, [
                "onnx", str(tmp_model),
                "--output-dir", str(out_dir),
                "--no-simplify",
            ])

        assert result.exit_code == 0, result.output
        assert "complete" in result.output.lower() or "onnx" in result.output.lower()

    def test_tensorrt_command_missing_model_exits_1(self, tmp_path):
        result = runner.invoke(convert_app, [
            "tensorrt", str(tmp_path / "missing.pt"),
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 1

    def test_tensorrt_command_trt_absent_succeeds_with_onnx(self, tmp_model, tmp_path):
        """When TRT is absent, pipeline completes with engine_path=None."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        onnx_out = out_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx")

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = str(onnx_out)

        # Simulate TRT absent by removing it from sys.modules for this call
        real_trt = sys.modules.get("tensorrt")
        sys.modules["tensorrt"] = None  # type: ignore[assignment]
        try:
            with patch("inference.tensorrt_engine.exporter.YOLO", return_value=mock_yolo_instance):
                result = runner.invoke(convert_app, [
                    "tensorrt", str(tmp_model),
                    "--output-dir", str(out_dir),
                    "--no-simplify",
                ])
        finally:
            if real_trt is None:
                sys.modules.pop("tensorrt", None)
            else:
                sys.modules["tensorrt"] = real_trt

        # Should exit 0 with a warning about TRT not available
        assert result.exit_code == 0, result.output

    def test_validate_command_missing_file_exits_1(self, tmp_path):
        result = runner.invoke(convert_app, [
            "validate", str(tmp_path / "nonexistent.onnx"),
        ])
        assert result.exit_code == 1

    def test_validate_command_passes_with_mock_ort(self, tmp_onnx, tmp_path):
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_input.shape = [1, 3, 1280, 1280]

        mock_output = np.zeros((1, 8, 8400), dtype=np.float32)
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [mock_output]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        with patch("inference.tensorrt_engine.exporter.ort", mock_ort):
            result = runner.invoke(convert_app, ["validate", str(tmp_onnx)])

        assert result.exit_code == 0, result.output
        assert "PASS" in result.output or "valid" in result.output.lower()

    def test_validate_command_fails_with_bad_onnx(self, tmp_onnx, tmp_path):
        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("bad model")

        with patch("inference.tensorrt_engine.exporter.ort", mock_ort):
            result = runner.invoke(convert_app, ["validate", str(tmp_onnx)])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _mock_import_without(blocked_module: str):
    """
    Return an __import__ side effect that raises ImportError for
    *blocked_module* and falls through to the real importer otherwise.
    """
    real_import = builtins_import  # captured before patching

    def _import(name, *args, **kwargs):
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"Mocked: {name} is not installed")
        return real_import(name, *args, **kwargs)

    return _import


# Capture real __import__ before any patching occurs
import builtins as _builtins_mod
builtins_import = _builtins_mod.__import__
