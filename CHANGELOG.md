# Changelog

All notable changes to CTIP. The project follows [Semantic Versioning](https://semver.org/); while in alpha, minor
versions may break APIs.

## [0.1.0-alpha] — 2026-10-01

First public release.

### Added
- Session-grouped dataset splits for the Label Studio export (leakage prevention), with warnings when sessions are
  missing or too few
- CI: CPU unit tests and frontend build on every push and pull request
- CONTRIBUTING, SECURITY, CITATION, issue and PR templates
- CTIP colour design for the web UI (light by default, dark variant)

### Changed
- Licence: AGPL-3.0-or-later (LICENSE text added; README badge and package metadata were inconsistent)
- README is a short front page; full EN/DE/ES manuals live in `docs/manual/`
- CLI and UI name: CTIP (was TrichomeLab)

### Fixed
- Fresh installs: missing `sqlmodel`/`alembic` dependencies, wheel package list, task store schema on first use
- TensorRT helpers report missing files before missing TensorRT
- 20 unit tests that only passed with an implicit asyncio event loop
- Frontend production build (ESLint)
- Documentation of CLI commands that did not exist

### Removed
- Local databases, logs, MLflow runs and environment files from the repository
