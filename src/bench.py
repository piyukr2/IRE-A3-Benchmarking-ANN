"""Code to perform benchmarking. Vary a search-parameter through a set of possible values and note down the observed recall and speed."""
import numpy as np

import config
from indexes import size_bytes
from metrics import median_qps, recall_at_k, single_thread


def sweep(index, data, set_param, param_values, info):
    """For each value of the search parameter, measure:
       - recall@10 on the tuning queries and on the final queries (batch search, all threads)
       - queries/second on the final queries, ONE query per call on ONE thread (median of repeats)
    `info` is a dict of columns (dataset, index name, build settings, build time ...) copied into every row.
    Stops early once the tuning recall is high enough, to save time."""
    rows = []
    for value in param_values:
        set_param(index, value)
        r_tune = recall_at_k(index.search(data.tune_q, config.K)[1], *data.gt_tune)
        r_final = recall_at_k(index.search(data.final_q, config.K)[1], *data.gt_final)
        with single_thread():
            qps = median_qps(lambda q: index.search(q, config.K),
                             data.final_q[: config.N_TIME], config.REPEATS)
        rows.append({**info, "search_param": value,
                     "recall_tune": r_tune, "recall": r_final, "qps": qps})
        print(f"  {info['index']:9s} {info['build']:28s} param={value:<5} "
              f"recall tune={r_tune:.3f} final={r_final:.3f}  qps={qps:,.0f}", flush=True)
        if r_tune >= config.STOP_RECALL:
            break
    return rows


def build_info(data, name, build, index, build_s):
    """Columns describing one built index."""
    size = size_bytes(index)
    return {"dataset": data.name, "n_train": len(data.train), "index": name, "build": build,
            "build_s": build_s, "size_mb": size / 1e6, "bytes_per_vec": size / len(data.train)}
