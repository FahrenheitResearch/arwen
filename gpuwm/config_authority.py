"""Immutable configuration-byte authority shared by CLI process phases.

Ordinary invocations read the requested source path.  A multi-run child is
instead given a create-only captured payload plus the digest and original
source path in its environment.  Every loader then consumes the captured
bytes, while the source path remains the diagnostic identity and relative
path base only.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CONFIG_PAYLOAD_ENV = "GPUWM_CONFIG_AUTHORITY_PAYLOAD"
CONFIG_SOURCE_ENV = "GPUWM_CONFIG_AUTHORITY_SOURCE"
CONFIG_SHA256_ENV = "GPUWM_CONFIG_AUTHORITY_SHA256"


@dataclass(frozen=True)
class ConfigAuthority:
    """One verified config payload and its immutable source semantics."""

    payload: bytes
    source: Path
    sha256: str
    payload_path: Path

    @property
    def base_dir(self) -> Path:
        return self.source.parent


def authority_environment(*, source: str | Path, payload_path: str | Path,
                          sha256: str) -> dict[str, str]:
    """Environment fields that bind loaders to a captured payload."""

    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError(f"config authority SHA-256 is invalid: {sha256!r}")
    return {
        CONFIG_PAYLOAD_ENV: str(Path(payload_path).resolve()),
        CONFIG_SOURCE_ENV: str(Path(source).resolve()),
        CONFIG_SHA256_ENV: sha256,
    }


def read_config_authority(
        path: str | Path, *, environment: Mapping[str, str] | None = None,
        ) -> ConfigAuthority:
    """Read either the ordinary source or its environment-bound capture.

    A partially configured authority is an error.  The supplied path must
    match the bound original source exactly, preventing the process-wide
    environment from silently redirecting a second unrelated config.
    """

    source = Path(path).expanduser().resolve()
    env = os.environ if environment is None else environment
    values = {
        CONFIG_PAYLOAD_ENV: env.get(CONFIG_PAYLOAD_ENV),
        CONFIG_SOURCE_ENV: env.get(CONFIG_SOURCE_ENV),
        CONFIG_SHA256_ENV: env.get(CONFIG_SHA256_ENV),
    }
    configured = [name for name, value in values.items() if value]
    if configured and len(configured) != len(values):
        missing = sorted(name for name, value in values.items() if not value)
        raise RuntimeError(
            "config authority environment is incomplete; missing "
            f"{missing}")
    if not configured:
        # Ordinary invocation: this is the ONE place a configuration is
        # read off disk, so it is where the path's KIND is decided.
        # `read_bytes()` has no opinion about what it is reading -- a
        # directory arrives as IsADirectoryError, a zero-byte file as an
        # eight-argument `RunConfig.__init__()` TypeError further down,
        # and a FIFO never returns at all.  Guarding here rather than at
        # each caller is deliberate: there are six call sites and the
        # next one added would otherwise inherit the old behaviour.
        #
        # The environment-bound branch above is NOT guarded, because a
        # multi-run child is handed the ORIGINAL source path purely as
        # the identity its captured payload is bound to; requiring that
        # path to still be a readable file on the child would refuse a
        # legitimate run.
        from gpuwm.experiment import readable_config_path

        source = readable_config_path(source)
        payload = source.read_bytes()
        return ConfigAuthority(
            payload, source, hashlib.sha256(payload).hexdigest(), source)

    bound_source = Path(values[CONFIG_SOURCE_ENV]).expanduser().resolve()
    if bound_source != source:
        raise RuntimeError(
            f"config authority is bound to {bound_source}, not requested "
            f"source {source}")
    expected = values[CONFIG_SHA256_ENV]
    if (len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise RuntimeError(
            f"config authority expected SHA-256 is invalid: {expected!r}")
    payload_path = Path(values[CONFIG_PAYLOAD_ENV]).expanduser().resolve()
    payload = payload_path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise RuntimeError(
            "captured config authority digest mismatch: expected "
            f"{expected}, observed {observed} at {payload_path}")
    return ConfigAuthority(payload, source, observed, payload_path)


__all__ = [
    "CONFIG_PAYLOAD_ENV", "CONFIG_SHA256_ENV", "CONFIG_SOURCE_ENV",
    "ConfigAuthority", "authority_environment", "read_config_authority",
]
