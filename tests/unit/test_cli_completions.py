"""
tests/unit/test_cli_completions.py — Unit tests for apps.cli.commands.completions.

Tests cover:
  - Shell auto-detection logic (_detect_shell)
  - Script generation (mocked subprocess)
  - install command: all 3 shells + auto + invalid + dry-run
  - show command: all 3 shells + invalid
  - uninstall command: file exists / missing / auto / dry-run
  - RC file patching helpers
  - Edge cases: unknown shell env, missing $SHELL
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from typer.testing import CliRunner

# Module under test
from apps.cli.commands.completions import (
    app,
    _detect_shell,
    _generate_completion_script,
    _resolve_target,
    _rc_file,
    _patch_rc_file,
    _SUPPORTED_SHELLS,
    _TYPER_COMPLETE_VAR,
    _INSTALL_TARGETS,
)

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_BASH_SCRIPT = "_TRICHOME_COMPLETE=bash_source\n# bash completion"
FAKE_ZSH_SCRIPT  = "#compdef trichome\n# zsh completion"
FAKE_FISH_SCRIPT = "complete -c trichome\n# fish completion"

_FAKE_SCRIPTS = {
    "bash": FAKE_BASH_SCRIPT,
    "zsh":  FAKE_ZSH_SCRIPT,
    "fish": FAKE_FISH_SCRIPT,
}


def _mock_generate(shell: str, cli_name: str = "trichome") -> str:
    """Return a fake completion script for the given shell."""
    return _FAKE_SCRIPTS[shell]


# ---------------------------------------------------------------------------
# _detect_shell
# ---------------------------------------------------------------------------

class TestDetectShell:
    def test_detects_bash_from_shell_env(self):
        with patch.dict(os.environ, {"SHELL": "/bin/bash"}):
            assert _detect_shell() == "bash"

    def test_detects_zsh_from_shell_env(self):
        with patch.dict(os.environ, {"SHELL": "/usr/bin/zsh"}):
            assert _detect_shell() == "zsh"

    def test_detects_fish_from_shell_env(self):
        with patch.dict(os.environ, {"SHELL": "/usr/local/bin/fish"}):
            assert _detect_shell() == "fish"

    def test_unknown_shell_env_returns_unknown(self):
        env = {k: v for k, v in os.environ.items() if k != "SHELL"}
        env["SHELL"] = "/usr/bin/tcsh"
        with patch.dict(os.environ, env, clear=True):
            # tcsh is not supported — may fall back to proc or return "unknown"
            result = _detect_shell()
            assert result in list(_SUPPORTED_SHELLS) + ["unknown"]

    def test_missing_shell_env_returns_unknown_or_detects(self):
        env = {k: v for k, v in os.environ.items() if k not in ("SHELL", "_")}
        with patch.dict(os.environ, env, clear=True):
            result = _detect_shell()
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _generate_completion_script
# ---------------------------------------------------------------------------

class TestGenerateCompletionScript:
    def test_returns_string_for_bash(self):
        with patch(
            "apps.cli.commands.completions.subprocess.run",
            return_value=MagicMock(returncode=0, stdout=FAKE_BASH_SCRIPT, stderr=""),
        ):
            script = _generate_completion_script("bash")
        assert isinstance(script, str)
        assert len(script) > 0

    def test_returns_string_for_zsh(self):
        with patch(
            "apps.cli.commands.completions.subprocess.run",
            return_value=MagicMock(returncode=0, stdout=FAKE_ZSH_SCRIPT, stderr=""),
        ):
            script = _generate_completion_script("zsh")
        assert "#compdef" in script or len(script) > 0

    def test_returns_string_for_fish(self):
        with patch(
            "apps.cli.commands.completions.subprocess.run",
            return_value=MagicMock(returncode=0, stdout=FAKE_FISH_SCRIPT, stderr=""),
        ):
            script = _generate_completion_script("fish")
        assert len(script) > 0

    def test_fallback_on_empty_stdout(self):
        """First call returns empty stdout; second (python fallback) returns real script."""
        side_effects = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=FAKE_BASH_SCRIPT, stderr=""),
        ]
        with patch(
            "apps.cli.commands.completions.subprocess.run",
            side_effect=side_effects,
        ):
            script = _generate_completion_script("bash")
        assert script == FAKE_BASH_SCRIPT

    def test_raises_if_both_calls_fail(self):
        empty = MagicMock(returncode=1, stdout="", stderr="error occurred")
        with patch(
            "apps.cli.commands.completions.subprocess.run",
            return_value=empty,
        ):
            with pytest.raises(RuntimeError, match="Failed to generate"):
                _generate_completion_script("bash")


# ---------------------------------------------------------------------------
# install command
# ---------------------------------------------------------------------------

class TestInstallCommand:
    def test_install_bash(self, tmp_path):
        target = tmp_path / ".bash_completion.d" / "trichome"
        rc_file = tmp_path / ".bashrc"
        rc_file.write_text("# existing bashrc\n")

        with (
            patch("apps.cli.commands.completions._generate_completion_script", return_value=FAKE_BASH_SCRIPT),
            patch("apps.cli.commands.completions._resolve_target", return_value=target),
            patch("apps.cli.commands.completions._patch_rc_file"),
        ):
            result = runner.invoke(app, ["install", "bash"])

        assert result.exit_code == 0, result.output
        assert target.exists()
        assert target.read_text() == FAKE_BASH_SCRIPT

    def test_install_zsh(self, tmp_path):
        target = tmp_path / ".zsh" / "completions" / "_trichome"

        with (
            patch("apps.cli.commands.completions._generate_completion_script", return_value=FAKE_ZSH_SCRIPT),
            patch("apps.cli.commands.completions._resolve_target", return_value=target),
            patch("apps.cli.commands.completions._patch_rc_file"),
        ):
            result = runner.invoke(app, ["install", "zsh"])

        assert result.exit_code == 0, result.output
        assert target.exists()
        assert target.read_text() == FAKE_ZSH_SCRIPT

    def test_install_fish(self, tmp_path):
        target = tmp_path / ".config" / "fish" / "completions" / "trichome.fish"

        with (
            patch("apps.cli.commands.completions._generate_completion_script", return_value=FAKE_FISH_SCRIPT),
            patch("apps.cli.commands.completions._resolve_target", return_value=target),
            patch("apps.cli.commands.completions._patch_rc_file"),
        ):
            result = runner.invoke(app, ["install", "fish"])

        assert result.exit_code == 0, result.output
        assert target.exists()

    def test_install_auto_detects_shell(self, tmp_path):
        target = tmp_path / ".bash_completion.d" / "trichome"

        with (
            patch("apps.cli.commands.completions._detect_shell", return_value="bash"),
            patch("apps.cli.commands.completions._generate_completion_script", return_value=FAKE_BASH_SCRIPT),
            patch("apps.cli.commands.completions._resolve_target", return_value=target),
            patch("apps.cli.commands.completions._patch_rc_file"),
        ):
            result = runner.invoke(app, ["install"])  # no shell arg → "auto"

        assert result.exit_code == 0, result.output

    def test_install_auto_unknown_shell_exits_1(self):
        with patch("apps.cli.commands.completions._detect_shell", return_value="unknown"):
            result = runner.invoke(app, ["install"])
        assert result.exit_code == 1
        assert "detect" in result.output.lower() or "auto" in result.output.lower()

    def test_install_unsupported_shell_exits_1(self):
        result = runner.invoke(app, ["install", "powershell"])
        assert result.exit_code == 1
        assert "Unsupported" in result.output or "unsupported" in result.output

    def test_install_dry_run_does_not_write(self, tmp_path):
        target = tmp_path / ".bash_completion.d" / "trichome"

        with (
            patch("apps.cli.commands.completions._generate_completion_script", return_value=FAKE_BASH_SCRIPT),
            patch("apps.cli.commands.completions._resolve_target", return_value=target),
        ):
            result = runner.invoke(app, ["install", "bash", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert not target.exists(), "dry-run must not create files"
        assert "would write" in result.output.lower() or "dry" in result.output.lower()

    def test_install_generation_failure_exits_1(self):
        with (
            patch(
                "apps.cli.commands.completions._generate_completion_script",
                side_effect=RuntimeError("subprocess failed"),
            ),
        ):
            result = runner.invoke(app, ["install", "bash"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# show command
# ---------------------------------------------------------------------------

class TestShowCommand:
    def test_show_bash(self):
        with patch(
            "apps.cli.commands.completions._generate_completion_script",
            return_value=FAKE_BASH_SCRIPT,
        ):
            result = runner.invoke(app, ["show", "bash"])
        assert result.exit_code == 0, result.output
        assert FAKE_BASH_SCRIPT in result.output

    def test_show_zsh(self):
        with patch(
            "apps.cli.commands.completions._generate_completion_script",
            return_value=FAKE_ZSH_SCRIPT,
        ):
            result = runner.invoke(app, ["show", "zsh"])
        assert result.exit_code == 0, result.output
        assert FAKE_ZSH_SCRIPT in result.output

    def test_show_fish(self):
        with patch(
            "apps.cli.commands.completions._generate_completion_script",
            return_value=FAKE_FISH_SCRIPT,
        ):
            result = runner.invoke(app, ["show", "fish"])
        assert result.exit_code == 0, result.output
        assert FAKE_FISH_SCRIPT in result.output

    def test_show_unsupported_shell_exits_1(self):
        result = runner.invoke(app, ["show", "csh"])
        assert result.exit_code == 1

    def test_show_generation_failure_exits_1(self):
        with patch(
            "apps.cli.commands.completions._generate_completion_script",
            side_effect=RuntimeError("generation error"),
        ):
            result = runner.invoke(app, ["show", "bash"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# uninstall command
# ---------------------------------------------------------------------------

class TestUninstallCommand:
    def test_uninstall_removes_existing_file(self, tmp_path):
        target = tmp_path / "trichome"
        target.write_text(FAKE_BASH_SCRIPT)
        assert target.exists()

        with patch("apps.cli.commands.completions._resolve_target", return_value=target):
            result = runner.invoke(app, ["uninstall", "bash"])

        assert result.exit_code == 0, result.output
        assert not target.exists()
        assert "Removed" in result.output

    def test_uninstall_missing_file_warns_gracefully(self, tmp_path):
        target = tmp_path / "nonexistent_trichome"

        with patch("apps.cli.commands.completions._resolve_target", return_value=target):
            result = runner.invoke(app, ["uninstall", "bash"])

        assert result.exit_code == 0, result.output
        # should not crash; should report nothing to remove
        assert "not found" in result.output.lower() or "already" in result.output.lower()

    def test_uninstall_auto_detects_shell(self, tmp_path):
        target = tmp_path / "trichome_fish"
        target.write_text(FAKE_FISH_SCRIPT)

        with (
            patch("apps.cli.commands.completions._detect_shell", return_value="fish"),
            patch("apps.cli.commands.completions._resolve_target", return_value=target),
        ):
            result = runner.invoke(app, ["uninstall"])

        assert result.exit_code == 0, result.output
        assert not target.exists()

    def test_uninstall_auto_unknown_shell_exits_1(self):
        with patch("apps.cli.commands.completions._detect_shell", return_value="unknown"):
            result = runner.invoke(app, ["uninstall"])
        assert result.exit_code == 1

    def test_uninstall_unsupported_shell_exits_1(self):
        result = runner.invoke(app, ["uninstall", "ksh"])
        assert result.exit_code == 1

    def test_uninstall_dry_run_does_not_delete(self, tmp_path):
        target = tmp_path / "trichome"
        target.write_text(FAKE_BASH_SCRIPT)

        with patch("apps.cli.commands.completions._resolve_target", return_value=target):
            result = runner.invoke(app, ["uninstall", "bash", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert target.exists(), "dry-run must not delete files"
        assert "would delete" in result.output.lower() or "dry" in result.output.lower()


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_resolve_target_bash(self):
        path = _resolve_target("bash")
        assert path.name == "trichome"
        assert ".bash_completion.d" in str(path)

    def test_resolve_target_zsh(self):
        path = _resolve_target("zsh")
        assert path.name == "_trichome"
        assert "completions" in str(path)

    def test_resolve_target_fish(self):
        path = _resolve_target("fish")
        assert path.suffix == ".fish"
        assert "fish" in str(path)

    def test_rc_file_bash(self):
        assert _rc_file("bash") == ".bashrc"

    def test_rc_file_zsh(self):
        assert _rc_file("zsh") == ".zshrc"

    def test_rc_file_fish(self):
        assert "fish" in _rc_file("fish")

    def test_supported_shells_constant(self):
        assert "bash" in _SUPPORTED_SHELLS
        assert "zsh" in _SUPPORTED_SHELLS
        assert "fish" in _SUPPORTED_SHELLS

    def test_typer_complete_var_keys(self):
        for shell in _SUPPORTED_SHELLS:
            assert shell in _TYPER_COMPLETE_VAR
            assert "_source" in _TYPER_COMPLETE_VAR[shell]

    def test_install_targets_keys(self):
        for shell in _SUPPORTED_SHELLS:
            assert shell in _INSTALL_TARGETS

    def test_patch_rc_bash_appends_snippet(self, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text("# existing\n")

        with (
            patch("apps.cli.commands.completions.Path.expanduser", return_value=rc),
        ):
            _patch_rc_file("bash")

        content = rc.read_text()
        assert "trichome" in content or "bash_completion" in content

    def test_patch_rc_fish_no_op(self, tmp_path):
        """Fish doesn't need rc patching — function should return without writing."""
        rc = tmp_path / "config.fish"
        rc.write_text("# fish config\n")
        original = rc.read_text()

        with patch("apps.cli.commands.completions.Path.expanduser", return_value=rc):
            _patch_rc_file("fish")

        # Fish rc should be unchanged (function returns early)
        assert rc.read_text() == original
