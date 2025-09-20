"""
Small, importable helpers for the trabajoFM pipeline.

Modules:
- io_fig: read/modify FIG data
- recyear_writer: write RECYEAR-related files
- std_parser: context-aware .std parser(s)
- graphs: network graphs/reducers
- runner: run_swat_model entry
- realizations: batch linking + runs across realizations
 - realizations: batch linking + runs across realizations
 - provenance: simple JSONL ledger + step logging
 - realization_id: monotonic ID allocator
 - mc_spec: experiment spec dataclasses
- mc_engine: Monte Carlo orchestration
 - mc_engine: Monte Carlo orchestration
 - writers.chm_writer: write/modify CHM files
 - transforms.soil_chm: deterministic conversions and MC transform for CHM
- rch_parser: read and tidy SWAT output.rch
- comparisons: utilities to compare DataFrames
- utils: shared helpers (config, logging, paths)
"""

from .spec_runner import run_from_spec

__all__ = [
    "io_fig",
    "recyear_writer",
    "std_parser",
    "graphs",
    "runner",
    "realizations",
    "provenance",
    "realization_id",
    "mc_spec",
    "mc_engine",
    "spec_runner",
    "writers",
    "transforms",
    "rch_parser",
    "comparisons",
    "utils",
    "run_from_spec",
    "write_provenance_reports"
]

__version__ = "0.1.0"
