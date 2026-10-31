# Hypotheses

## Part 3: Compression (SIFT, 1M vectors)

Setup: IVF-PQ with m = 8, 16, 32, 64 sub-vectors, plain vs. followed by exact re-ranking
of the top 100 candidates (`IndexRefineFlat`, k_factor = 10).

**Hypothesis.**
1) IVF-PQ squeezes each vector into a fewer dimensions, so some information is permanentl lost. Its recall will stop rising at some ceiling however many cells I search. More sub-vectors (i.e.larger m) means a higher ceiling.
2) Re-ranking looks up the full vectors for the top 100 candidates and sorts them exactly. I expect it to lift recall above 0.95 even for small m.
3) The cost is memory. Re-ranking must keep the full 512-byte vectors, so memory will be about 512 + m bytes per vector, against m bytes for plain IVF-PQ.

## Part 4: Near-duplicates (hard geometry)

Setup: 50,000 real SIFT vectors, each present with 1 (control), 2, 5 or 10 copies
(copies = original + Gaussian noise, sigma = 1 per coordinate); total index size up to 500,000.
Compared on HNSW (M = 16) and IVF-Flat (nlist = 256).

**Hypothesis.**
- HNSW will do worse as each vector gets more copies. Copies fill each point's few edges, so the search stays stuck among copies instead of reaching the right area. It will need a larger efSearch for the same recall.
- IVF-Flat should stay almost unchanged. A vector and its copies fall in the same cell, so the same nprobe still finds them. It will only slow down in proportion to the larger index.
- If instead HNSW is unaffected, the most probable reason is its neighbour-selection heuristic, which avoids keeping edges to points that are closer to an existing neighbour than to the node.
