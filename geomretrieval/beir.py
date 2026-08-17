from __future__ import annotations
import csv
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BEIRDataset:
    corpus_ids: list[str]
    corpus_texts: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, float]]
    name: str = "dataset"


def _find(root: Path, basename: str) -> Path:
    hits = list(root.rglob(basename))
    if not hits:
        raise FileNotFoundError(f"Could not find {basename!r} under {root}")
    if len(hits) > 1:
        # Prefer the shallowest path, which is normally the dataset root.
        hits.sort(key=lambda p: len(p.parts))
    return hits[0]


def load_beir_directory(path: str | os.PathLike, split: str = "test") -> BEIRDataset:
    root = Path(path)
    corpus_path = _find(root, "corpus.jsonl")
    queries_path = _find(root, "queries.jsonl")
    qrels_hits = list(root.rglob(f"qrels/{split}.tsv"))
    if not qrels_hits:
        # Some archives flatten qrels paths.
        qrels_hits = [p for p in root.rglob(f"{split}.tsv") if p.parent.name == "qrels"]
    if not qrels_hits:
        raise FileNotFoundError(f"Could not find qrels/{split}.tsv under {root}")
    qrels_path = qrels_hits[0]

    corpus_ids, corpus_texts = [], []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = str(obj.get("_id", obj.get("id")))
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            merged = (title + " " + text).strip()
            corpus_ids.append(did)
            corpus_texts.append(merged)

    queries = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id", obj.get("id")))
            queries[qid] = obj.get("text", "") or ""

    qrels: dict[str, dict[str, float]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # BEIR normally uses query-id, corpus-id, score.
            qid = str(row.get("query-id", row.get("query_id", row.get("qid"))))
            did = str(row.get("corpus-id", row.get("corpus_id", row.get("docid"))))
            score = float(row.get("score", row.get("relevance", row.get("rel", 0))))
            qrels.setdefault(qid, {})[did] = score

    name = corpus_path.parent.name
    return BEIRDataset(corpus_ids, corpus_texts, queries, qrels, name=name)


def load_beir_zip(path: str | os.PathLike, split: str = "test") -> BEIRDataset:
    """Load a standard BEIR zip without requiring internet access."""
    with tempfile.TemporaryDirectory(prefix="geomretrieval_beir_") as td:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(td)
        return load_beir_directory(td, split=split)
