# Clean-runner verification

These files were produced after the repository was assembled, using the cleaned path-independent runner against the already-built SciFact and TREC-COVID indexes.

They verify that the cleaned runner reproduces the stored P=100 effectiveness values. Its `median_total_ms` is a **correctness/reference Python timing**, not the optimized speed target; current top-10 branch-aware `rank_ms` is reported separately, and the optimized full-scale implementations remain under `experiments/msmarco_scale/`.
