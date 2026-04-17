"""Data module for pipeline v3: VitalDB acquisition, case catalogue, loaders."""

from .vitaldb_downloader import (
    DEFAULT_BIS_TRACK,
    DEFAULT_EEG_TRACK,
    CatalogueColumns,
    QCThresholds,
    SelectionSpec,
    build_catalogue,
    download_and_save_case,
    run_download,
    select_case_ids,
)

__all__ = [
    "DEFAULT_EEG_TRACK",
    "DEFAULT_BIS_TRACK",
    "CatalogueColumns",
    "QCThresholds",
    "SelectionSpec",
    "select_case_ids",
    "download_and_save_case",
    "build_catalogue",
    "run_download",
]
