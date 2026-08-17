"""Frozen sparse geometric retrieval package."""
from .config import FrozenConfig
from .index import GeometricIndex
from .beir import load_beir_zip, load_beir_directory
from .metrics import evaluate_run

__all__ = ["FrozenConfig", "GeometricIndex", "load_beir_zip", "load_beir_directory", "evaluate_run"]
__version__ = "0.1.0"

from .rag_top10 import RAGTop10Config, RAGTop10Ranker
