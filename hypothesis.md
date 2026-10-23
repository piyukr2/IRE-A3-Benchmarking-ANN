# Hypotheses (written BEFORE running the experiments)

> DRAFT - read, rewrite in your own words, and commit before running Parts 3 and 4.

## Part 3: Compression (SIFT, 1M vectors)

Setup: IVF-PQ with m = 8, 16, 32, 64 sub-vectors, plain vs. followed by exact re-ranking
of the top 100 candidates (`IndexRefineFlat`, k_factor = 10).

**Hypothesis.**
1. Plain IVF-PQ recall stops growing at some ceiling no matter how large nprobe is, because
   the PQ codes lose information; the ceiling rises with m (more bytes per vector = finer
   approximation) but stays below 0.95 for small m (8, 16).
2. Re-ranking removes that ceiling: PQ only has to put the true neighbours somewhere in the
   top 100, and exact distances then sort them correctly. Recall should reach 0.95+ even for small m.
3. The price is memory: re-ranking needs the full vectors (512 bytes each on SIFT), so the
   memory of the refined index is about 512 bytes + m bytes per vector, almost independent of m,
   versus only m bytes (plus overhead) for plain IVF-PQ. Speed drops a little because of the extra
   100 exact distance computations per query.

## Part 4: Near-duplicates (hard geometry)

Setup: 50,000 real SIFT vectors, each present with 1 (control), 2, 5 or 10 copies
(copies = original + Gaussian noise, sigma = 1 per coordinate); total index size up to 500,000.
Compared on HNSW (M = 16) and IVF-Flat (nlist = 256).

**Hypothesis.**
- HNSW will lose recall at a fixed efSearch as the number of copies grows. A node's M edges
  get used up on its own near-identical copies, so the graph has fewer long-range edges and the
  search gets stuck inside a clique of copies instead of travelling to the right region.
  I expect it to need a much larger efSearch to reach the same recall.
- IVF-Flat will barely be affected: k-means puts a vector and its copies in the same cell, so
  a query still reaches the right cell with the same nprobe; the cost per probe only grows
  because cells hold more vectors (QPS falls roughly in proportion to the index size).
- If instead HNSW is unaffected, the likely reason is its neighbour-selection heuristic, which
  avoids keeping edges to points that are closer to an existing neighbour than to the node.
