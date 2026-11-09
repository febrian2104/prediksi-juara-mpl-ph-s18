"""Canonical MPL PH datasets, organized like the MPL Indonesia reference."""

from mpl_ph_predictor.data.normalization import (
    build_canonical_data,
    compute_standings,
    load_identity_map,
    slug,
    write_canonical_data,
)

__all__ = [
    "build_canonical_data",
    "compute_standings",
    "load_identity_map",
    "slug",
    "write_canonical_data",
]
