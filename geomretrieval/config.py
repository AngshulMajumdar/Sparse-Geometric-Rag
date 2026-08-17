from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class FrozenConfig:
    """Frozen MS-MARCO-developed configuration.

    The point of the six-dataset campaign is transfer, not per-dataset tuning.
    Only corpus-mechanical settings such as max_features/min_df should be changed
    when a dataset physically requires it.
    """

    # Sparse lexical representation
    max_features: int = 50_000
    min_df: int = 1
    lowercase: bool = True
    token_pattern: str = r"(?u)\b\w\w+\b"

    # Fuzzy index
    F: int = 4                   # fuzzy memberships/document
    B: int = 64                  # sparse center support
    S: int = 16                  # signed residual support/membership

    # Reliability
    tau: float = 20.0
    beta: float = -0.2
    reliability_eps: float = 1e-6

    # Corpus term geometry
    L: int = 12                  # top terms/document used to estimate graph
    graph_significance_tau: float = 10.0
    assoc_k: int = 64            # first-order PPMI neighbors retained
    route_k: int = 32            # second-order context neighbors retained
    graph_block_size: int = 128

    # Query routing
    route_alpha: float = 0.10
    route_budget: int = 32       # strongest total route coordinates; original terms preserved

    # Head / tail scoring
    head_k: int = 10
    gamma_head: float = 0.5
    gamma_tail: float = 1.0
    lambda_membership: float = 2.0

    # Final binary-support reranker
    rerank_pool: int = 2_000
    lambda_lex: float = 2.5
    length_b: float = 0.2
    semantic_k: int = 16
    lambda_sem: float = 0.05

    # Requested output depth
    output_k: int = 100

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenConfig":
        return cls(**d)
