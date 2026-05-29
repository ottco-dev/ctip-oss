"""
apps.cli.commands.completions — Shell completion script management.

Provides a dedicated ``trichome completions`` subcommand group for generating,
installing, and removing shell completion scripts for bash, zsh, and fish.

Typer's built-in ``--install-completion`` / ``--show-completion`` flags work
via Click's completion machinery; this module wraps that mechanism into an
explicit, user-facing subcommand that also writes stand-alone script files
to the correct per-user locations.

Usage:
    trichome completions install          # auto-detect shell, install
    trichome completions install bash     # install for bash explicitly
    trichome completions install zsh
    trichome completions install fish
    trichome completions show bash        # print script to stdout
    trichome completions uninstall        # auto-detect and remove
    trichome completions uninstall zsh
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()

app = typer.Typer(
    name="completions",
    help="Manage shell completion scripts (bash / zsh / fish).",
    add_help_option=True,
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Supported shells
# ---------------------------------------------------------------------------

_SUPPORTED_SHELLS = ("bash", "zsh", "fish")

# Typer/Click completion env var values per shell
_TYPER_COMPLETE_VAR: dict[str, str] = {
    "bash": "bash_source",
    "zsh":  "zsh_source",
    "fish": "fish_source",
}

# Default installation targets (resolved relative to HOME at runtime)
_INSTALL_TARGETS: dict[str, str] = {
    "bash": "~/.bash_completion.d/trichome",
    "zsh":  "~/.zsh/completions/_trichome",
    "fish": "~/.config/fish/completions/trichome.fish",
}

# Lines injected into shell rc files when the completion directory approach
# isn't self-sourcing.
_BASH_SOURCE_SNIPPET = "\n# TrichomeLab CLI completions\n[ -f ~/.bash_completion.d/trichome ] && source ~/.bash_completion.d/trichome\n"
_ZSH_SOURCE_SNIPPET  = "\n# TrichomeLab CLI completions\nfpath=(~/.zsh/completions $fpath)\nautoload -Uz compinit && compinit\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_shell() -> str:
    """Return the name of the running user shell (bash | zsh | fish | unknown)."""
    # 1. $SHELL environment variable (most reliable on Linux/macOS)
    shell_env = os.environ.get("SHELL", "")
    if shell_env:
        shell_name = Path(shell_env).name.lower()
        if shell_name in _SUPPORTED_SHELLS:
            return shell_name

    # 2. Inspect the parent process name (works in some CI environments)
    try:
        ppid = os.getppid()
        proc_name_path = Path(f"/proc/{ppid}/comm")
        if proc_name_path.exists():
            name = proc_name_path.read_text().strip().lower()
            if name in _SUPPORTED_SHELLS:
                return name
    except Exception:
        pass

    # 3. Check $0 via os.environ
    zero = os.environ.get("_", "")
    if zero:
        name = Path(zero).name.lower()
        if name in _SUPPORTED_SHELLS:
            return name

    return "unknown"


def _generate_completion_script(shell: str, cli_name: str = "trichome") -> str:
    """
    Generate the completion script for *shell* by running the CLI process
    with the appropriate ``_<PROG>_COMPLETE`` env-var set.

    Returns the script as a string.  Raises ``RuntimeError`` on failure.
    """
    env_key = f"_{cli_name.upper().replace('-', '_')}_COMPLETE"
    env_val = _TYPER_COMPLETE_VAR[shell]
    env = {**os.environ, env_key: env_val}

    # The CLI entrypoint registered in pyproject.toml is ``trichome``
    # Use sys.executable + module path as a robust fallback in case the
    # ``trichome`` script isn't on PATH yet (e.g. during development).
    result = subprocess.run(
        [cli_name],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        # Fallback: run via python -m apps.cli.main
        result = subprocess.run(
            [sys.executable, "-c",
             f"from apps.cli.main import app; app()"],
            env=env,
            capture_output=True,
            text=True,
        )

    script = result.stdout.strip()
    if not script:
        raise RuntimeError(
            f"Failed to generate {shell} completion script. "
            f"stderr: {result.stderr.strip()}"
        )
    return script


def _resolve_target(shell: str) -> Path:
    """Return the absolute Path for the completion script installation target."""
    return Path(_INSTALL_TARGETS[shell]).expanduser()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("install")
def install(
    shell: str = typer.Argument(
        "auto",
        help="Target shell: bash | zsh | fish | auto  (auto detects from $SHELL)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be done without writing files."
    ),
) -> None:
    """
    Install shell completions for the current user.

    \b
    Bash:  writes to ~/.bash_completion.d/trichome
           and appends a source line to ~/.bashrc if needed.
    Zsh:   writes to ~/.zsh/completions/_trichome
           and appends fpath + compinit to ~/.zshrc if needed.
    Fish:  writes to ~/.config/fish/completions/trichome.fish
           (Fish picks this up automatically).

    \b
    Examples:
        trichome completions install
        trichome completions install bash
        trichome completions install fish --dry-run
    """
    resolved = shell.lower()

    if resolved == "auto":
        resolved = _detect_shell()
        if resolved == "unknown":
            console.print(
                "[red]Could not detect shell automatically.[/red] "
                "Pass the shell name explicitly: bash | zsh | fish"
            )
            raise typer.Exit(code=1)
        console.print(f"[dim]Detected shell:[/dim] [cyan]{resolved}[/cyan]")

    if resolved not in _SUPPORTED_SHELLS:
        console.print(
            f"[red]Unsupported shell:[/red] {resolved!r}. "
            f"Supported: {', '.join(_SUPPORTED_SHELLS)}"
        )
        raise typer.Exit(code=1)

    # Generate script
    console.print(f"[dim]Generating {resolved} completion script…[/dim]")
    try:
        script = _generate_completion_script(resolved)
    except RuntimeError as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        raise typer.Exit(code=1)

    target = _resolve_target(resolved)

    if dry_run:
        console.print(f"[yellow][dry-run][/yellow] Would write {len(script)} bytes to [cyan]{target}[/cyan]")
        _print_rc_instructions(resolved, dry_run=True)
        return

    # Write completion file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script)
    console.print(f"[green]Wrote completion script:[/green] {target}")

    # Patch shell rc file if needed
    _patch_rc_file(resolved)

    console.print()
    console.print(
        f"[bold green]Shell completions installed for {resolved}.[/bold green]  "
        f"Restart your shell or run: [dim]source ~/{_rc_file(resolved)}[/dim]"
    )


@app.command("show")
def show(
    shell: str = typer.Argument(
        "bash",
        help="Shell to generate for: bash | zsh | fish",
    ),
) -> None:
    """
    Print the completion script to stdout.

    Pipe the output into your shell config to activate completions:

    \b
        trichome completions show bash >> ~/.bashrc
        trichome completions show zsh  >> ~/.zshrc
        trichome completions show fish > ~/.config/fish/completions/trichome.fish

    \b
    Examples:
        trichome completions show bash
        trichome completions show zsh
    """
    resolved = shell.lower()
    if resolved not in _SUPPORTED_SHELLS:
        console.print(
            f"[red]Unsupported shell:[/red] {resolved!r}. "
            f"Supported: {', '.join(_SUPPORTED_SHELLS)}"
        )
        raise typer.Exit(code=1)

    try:
        script = _generate_completion_script(resolved)
    except RuntimeError as exc:
        console.print(f"[red]Failed to generate script:[/red] {exc}", file=sys.stderr)
        raise typer.Exit(code=1)

    # Write raw script to stdout (not through Rich, to keep it pipe-safe)
    sys.stdout.write(script + "\n")
    sys.stdout.flush()


@app.command("uninstall")
def uninstall(
    shell: str = typer.Argument(
        "auto",
        help="Shell to remove completions for: bash | zsh | fish | auto",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be removed without touching files."
    ),
) -> None:
    """
    Remove installed shell completions.

    \b
    Examples:
        trichome completions uninstall
        trichome completions uninstall zsh
        trichome completions uninstall bash --dry-run
    """
    resolved = shell.lower()

    if resolved == "auto":
        resolved = _detect_shell()
        if resolved == "unknown":
            console.print(
                "[red]Could not detect shell automatically.[/red] "
                "Pass the shell name explicitly."
            )
            raise typer.Exit(code=1)
        console.print(f"[dim]Detected shell:[/dim] [cyan]{resolved}[/cyan]")

    if resolved not in _SUPPORTED_SHELLS:
        console.print(
            f"[red]Unsupported shell:[/red] {resolved!r}. "
            f"Supported: {', '.join(_SUPPORTED_SHELLS)}"
        )
        raise typer.Exit(code=1)

    target = _resolve_target(resolved)

    if dry_run:
        if target.exists():
            console.print(f"[yellow][dry-run][/yellow] Would delete: [cyan]{target}[/cyan]")
        else:
            console.print(f"[yellow][dry-run][/yellow] File not found (nothing to remove): {target}")
        return

    if target.exists():
        target.unlink()
        console.print(f"[green]Removed:[/green] {target}")
    else:
        console.print(f"[yellow]Completion file not found (already removed?):[/yellow] {target}")

    console.print(
        f"[dim]Note: any source lines added to ~/{_rc_file(resolved)} were not removed "
        f"— remove manually if desired.[/dim]"
    )


# ---------------------------------------------------------------------------
# RC file helpers
# ---------------------------------------------------------------------------

def _rc_file(shell: str) -> str:
    """Return the primary rc filename for a shell (relative to HOME)."""
    return {
        "bash": ".bashrc",
        "zsh":  ".zshrc",
        "fish": ".config/fish/config.fish",
    }.get(shell, ".bashrc")


def _patch_rc_file(shell: str) -> None:
    """Append source/fpath lines to the shell rc file if not already present."""
    rc = Path(f"~/{_rc_file(shell)}").expanduser()

    if shell == "bash":
        snippet = _BASH_SOURCE_SNIPPET
        marker = "~/.bash_completion.d/trichome"
    elif shell == "zsh":
        snippet = _ZSH_SOURCE_SNIPPET
        marker = "~/.zsh/completions"
    else:
        # Fish auto-discovers ~/.config/fish/completions/ — no rc patching needed
        return

    existing = rc.read_text() if rc.exists() else ""
    if marker in existing:
        console.print(f"[dim]RC file already configured:[/dim] {rc}")
        return

    with open(rc, "a") as fh:
        fh.write(snippet)
    console.print(f"[dim]Updated:[/dim] {rc}")


def _print_rc_instructions(shell: str, *, dry_run: bool = False) -> None:
    """Print (without writing) the rc snippet that would be applied."""
    if shell == "bash":
        snippet = _BASH_SOURCE_SNIPPET.strip()
    elif shell == "zsh":
        snippet = _ZSH_SOURCE_SNIPPET.strip()
    else:
        return

    label = "[yellow][dry-run][/yellow] " if dry_run else ""
    console.print(f"{label}Would append to [cyan]~/{_rc_file(shell)}[/cyan]:")
    console.print(f"[dim]{snippet}[/dim]")


if __name__ == "__main__":
    app()
