"""Filesystem layout. All paths derive from ASTRA_ROOT (env) or the repository root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _find_repo_root() -> Path:
    env = os.environ.get("ASTRA_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config").exists():
            return parent
    return Path.cwd().resolve()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def data(self) -> Path:
        return Path(os.environ.get("ASTRA_DATA_DIR", self.root / "data"))

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def validated(self) -> Path:
        return self.data / "validated"

    @property
    def features(self) -> Path:
        return self.data / "features"

    @property
    def reports(self) -> Path:
        return self.data / "reports"

    @property
    def experiments(self) -> Path:
        return Path(os.environ.get("ASTRA_EXPERIMENTS_DIR", self.root / "experiments"))

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> Paths:
        for p in (self.raw, self.validated, self.features, self.reports, self.experiments, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: Path | None = None) -> Paths:
    return Paths(root=(root or _find_repo_root()).resolve())
