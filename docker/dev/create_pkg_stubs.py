"""
create_pkg_stubs.py — Docker build helper.

Reads pyproject.toml and creates empty __init__.py stubs for every package
listed under [tool.hatch.build.targets.wheel.packages].  This lets hatchling
build a valid (but empty) wheel during the dep-caching layer of Dockerfile.dev,
before the real source tree is copied in via `COPY . .`.
"""
import pathlib
import tomllib

with open("pyproject.toml", "rb") as f:
    cfg = tomllib.load(f)

pkgs: list[str] = (
    cfg.get("tool", {})
       .get("hatch", {})
       .get("build", {})
       .get("targets", {})
       .get("wheel", {})
       .get("packages", [])
)

created: list[str] = []
for p in pkgs:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    init = pathlib.Path(p) / "__init__.py"
    if not init.exists():
        init.write_text("# docker-dep-cache-stub — overwritten by COPY . .\n")
        created.append(str(init))

print(f"Stubs created ({len(created)}):")
for c in created:
    print(f"  {c}")
