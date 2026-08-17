# Method

## Offline representation

For each chunk, construct a sparse normalized TF-IDF vector. Keep its top `F=4` coordinates as fuzzy branch memberships. For branch `j`, keep a sparse `B=64` membership-weighted center. For every chunk-branch membership, retain only `S=16` residual coordinates chosen from terms actually present in the chunk and store their **signs**, not document-specific residual amplitudes.

A zero-inclusive branch-local sign variance is shrunk toward the global term variance and converted to a mild inverse reliability weight, approximately `variance^-0.2`. The corpus also produces a sparse PPMI association graph and a second-order context graph for query routing.

## Query routing and local scoring

The query retains real TF-IDF amplitudes. Weak second-order expansion exposes nearby branches. A branch-local signed score evaluates only the 16 stored residual coordinates and is weighted by fuzzy membership, route strength, query-mass significance, and inverse local sign variance.

## Early rescue and chunk shortlist

The routed representations are cheaply ordered using geometric evidence plus whole-chunk **binary IDF^1** support. The best `P` representations are mapped to their corresponding chunks. No dense 384/768-dimensional embedding is required at this stage.

## Final chunk score

For each shortlisted chunk, the final lexical statistic is binary presence weighted by **IDF squared**:

```text
sum_{t in query ∩ chunk} IDF(t)^2
```

It is combined with a weak length correction, query coordination, sparse semantic presence, and coverage of the three rarest query terms. The validated score components retain their per-query z-normalization.

## High-quality branches and soft diversity

Branch quality is query-specific. For branch `j`, define branch-specific evidence `E_dj` using the geometric membership contribution. The quality score is the mean of the top three evidences in that branch:

```text
H_j = mean(top3_d E_dj)
```

Only the ten branches with largest `H_j` are eligible for a diversity bonus. Rank 1 is pure relevance. For ranks 2–10, a candidate can receive a small bonus if one of its high-quality supporting branch centers deviates from the centroid of branches already represented. Repeated branches are allowed.

This is **not blind diversification** and it is **not one-document-per-branch**.
