"""Data module for pipeline v3: VitalDB acquisition, case catalogue, loaders."""

from .vitaldb_downloader import (
    DEFAULT_EEG_TRACK,
    DEFAULT_BIS_TRACK,
    CatalogueColumns,
    QCThresholds,
    SelectionSpec,
    select_case_ids,
    download_and_save_case,
    build_catalogue,
    run_download,
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
