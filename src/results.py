"""Append rows to results/results.csv. Re-running a part replaces that part's old rows."""
import pandas as pd

import config


def save_rows(part, rows):
    new = pd.DataFrame(rows)
    new.insert(0, "part", part)
    if config.RESULTS_CSV.exists():
        old = pd.read_csv(config.RESULTS_CSV)
        old = old[~old["part"].isin(new["part"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(config.RESULTS_CSV, index=False)


def load():
    return pd.read_csv(config.RESULTS_CSV)
