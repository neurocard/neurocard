#!/usr/bin/env python3

from functools import wraps
import os.path

import glog as log
import numpy as np
import pandas as pd

import datasets

CACHE_DIR = "./cache"

TOY_TABLES = {
    "A": pd.DataFrame({"x": [1, 2, 3]}),
    "B": pd.DataFrame({
        "x": [1, 2, 2, 2, 4],
        "y": [10, 20, 20, 30, 30],
        "z": [100, 100, 100, 100, 200],
    }),
    "C": pd.DataFrame({"y": [10, 20, 20, 40]}),
    "D": pd.DataFrame({"z": [100, 100, 200, 300]}),
}


def load(filename, description):
    save_path = os.path.join(CACHE_DIR, filename)
    log.info(f"Loading cached {description} from {save_path}")
    return pd.read_feather(save_path)


def save_result(filename, subdir=None, description="result"):

    def decorator(func):

        @wraps(func)
        def wrapper(*fargs, **kwargs):
            os.makedirs(CACHE_DIR, exist_ok=True)
            if subdir is not None:
                os.makedirs(os.path.join(CACHE_DIR, subdir), exist_ok=True)
                save_path = os.path.join(CACHE_DIR, subdir, filename)
            else:
                save_path = os.path.join(CACHE_DIR, filename)
            if os.path.exists(save_path):
                log.info(f"Loading cached {description} from {save_path}")
                ret = pd.read_feather(save_path)
            else:
                log.info(f"Creating {description}.")
                ret = func(*fargs, **kwargs)
                log.info(f"Saving {description} to {save_path}")
                ret.to_feather(save_path)
            return ret

        return wrapper

    return decorator


def load_table(table, data_dir="datasets/job", **kwargs):
    if table in TOY_TABLES:
        return TOY_TABLES[table]

    usecols = kwargs.get("usecols")
    if usecols == "job-m":
        usecols = datasets.JoinOrderBenchmark.JOB_M_PRED_COLS[f"{table}.csv"]
    kwargs.update({"usecols": usecols})
    if usecols is None:
        usecols = ["ALL"]

    @save_result("{}-{}.df".format(table, "-".join(usecols)),
                 description=f"dataframe of `{table}`")
    def work():
        print(table, kwargs)
        return pd.read_csv(os.path.join(data_dir, f"{table}.csv"),
                           escapechar="\\",
                           low_memory=False,
                           **kwargs)

    return work()
