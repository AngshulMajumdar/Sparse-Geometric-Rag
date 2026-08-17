from __future__ import annotations
import argparse
import json
from pathlib import Path

from .beir import load_beir_zip, load_beir_directory
from .config import FrozenConfig
from .index import GeometricIndex
from .metrics import evaluate_run


def _dataset(path: str, split: str):
    return load_beir_zip(path, split) if str(path).lower().endswith(".zip") else load_beir_directory(path, split)


def cmd_build(args):
    ds = _dataset(args.dataset, args.split)
    cfg = FrozenConfig(max_features=args.max_features, min_df=args.min_df)
    idx = GeometricIndex.build(ds.corpus_texts, ds.corpus_ids, cfg, verbose=True)
    idx.save(args.output)
    print(f"saved index -> {args.output}")


def cmd_eval(args):
    ds = _dataset(args.dataset, args.split)
    idx = GeometricIndex.load(args.index)
    # Evaluate only qrels-bearing queries.
    queries = {qid: ds.queries[qid] for qid in ds.qrels if qid in ds.queries}
    run, timing = idx.batch_search(queries, k=args.k, timing=True)
    metrics = evaluate_run(run, ds.qrels, ks=(10, 100), ndcg_k=10, mrr_k=10)
    out = {"dataset": ds.name, **metrics, **timing}
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.run_json:
        Path(args.run_json).write_text(json.dumps(run, indent=1))


def cmd_search(args):
    idx = GeometricIndex.load(args.index)
    ids, scores = idx.search(args.query, k=args.k, return_scores=True)
    for r, (d, s) in enumerate(zip(ids, scores), start=1):
        print(f"{r:3d}\t{d}\t{s:.6f}")


def main():
    p = argparse.ArgumentParser(prog="geomretrieval")
    sp = p.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build", help="Build frozen sparse index from a BEIR dataset/archive")
    b.add_argument("dataset")
    b.add_argument("output")
    b.add_argument("--split", default="test")
    b.add_argument("--max-features", type=int, default=50_000)
    b.add_argument("--min-df", type=int, default=1)
    b.set_defaults(func=cmd_build)

    e = sp.add_parser("eval", help="Evaluate an existing index on BEIR qrels")
    e.add_argument("dataset")
    e.add_argument("index")
    e.add_argument("--split", default="test")
    e.add_argument("--k", type=int, default=100)
    e.add_argument("--run-json", default=None)
    e.set_defaults(func=cmd_eval)

    s = sp.add_parser("search", help="Search an existing index")
    s.add_argument("index")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=10)
    s.set_defaults(func=cmd_search)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
