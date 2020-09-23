#!/usr/bin/env python3
"""Unbiased join sampler using the Exact Weight algorithm."""

import argparse
import collections
import os
import pickle
import time

import glog as log
import numpy as np
import pandas as pd

import common
import datasets
import experiments
import factorized_sampler_lib.data_utils as data_utils
import factorized_sampler_lib.prepare_utils as prepare_utils
import factorized_sampler_lib.rustlib as rustlib
import join_utils

# Assuming the join columns contain only non-negative values.
# TODO: Remove this assumption.
NULL = -1

# ----------------------------------------------------------------
#      Column names utils
# ----------------------------------------------------------------


def get_jct_count_columns(join_spec):
    return get_fanout_columns_impl(join_spec, "{table}.{key}.cnt",
                                   "{table}.{key}.cnt")


def get_fanout_columns(join_spec):
    return get_fanout_columns_impl(join_spec, "__fanout_{table}",
                                   "__fanout_{table}__{key}")


def get_fanout_columns_impl(join_spec, single_key_fmt, multi_key_fmt):
    ret = []
    for t in join_spec.join_tables:
        # TODO: support when join root needs to be downscaled.  E.g., Say the
        # tree looks like root -> A -> B. Now a query joins A join B. We still
        # want to downscale fanout introduced by root.
        if t == join_spec.join_root:
            continue
        keys = join_spec.join_keys[t]
        if len(keys) == 1:
            ret.append(single_key_fmt.format(table=t, key=keys[0]))
        else:
            for k in keys:
                ret.append(multi_key_fmt.format(table=t, key=k))
    return ret


# ----------------------------------------------------------------
#      Sampling from join count tables
# ----------------------------------------------------------------


def get_distribution(series):
    """Make a probability distribution out of a series of counts."""
    arr = series.values
    total = np.sum(arr)
    assert total > 0
    return arr / total


class JoinCountTableActor(object):

    def __init__(self, table, jct, join_spec):

        self.jct = jct
        self.table = table
        parents = list(join_spec.join_tree.predecessors(table))
        assert len(parents) <= 1, parents
        if len(parents) == 1:
            parent = parents[0]
            join_keys = join_spec.join_graph[parent][table]["join_keys"]
            self.table_join_key = f"{table}.{join_keys[table]}"
            self.parent_join_key = f"{parent}.{join_keys[parent]}"
            null_row_offset = self._insert_null_to_jct()
            self.index_provider = rustlib.IndexProvider(
                f"{join_spec.join_name}/{table}.jk.indices", null_row_offset)
        else:
            self.jct_distribution = get_distribution(
                self.jct[f"{self.table}.weight"])
        log.info(f"JoinCountTableActor `{table}` is ready.")

    def _insert_null_to_jct(self):
        # This is to simplify sampling.  NULL for every key column; 0 for the
        # count column to avoid being sampled.
        null_row = pd.Series(NULL, self.jct.columns)
        null_row[f"{self.table}.weight"] = 0
        null_row_offset = self.jct.shape[0]
        self.jct.loc[null_row_offset] = null_row
        return null_row_offset

    def take_sample(self, parent_sample, sample_size, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if parent_sample is None:
            indices = rng.choice(np.arange(self.jct.shape[0]),
                                 size=sample_size,
                                 replace=True,
                                 p=self.jct_distribution)
            sample = self.jct.iloc[indices].reset_index(drop=True)
            return sample

        keys = parent_sample[self.parent_join_key].values
        sample_indices = self.index_provider.sample_indices(keys)
        sample = self.jct.iloc[sample_indices].reset_index(drop=True)
        return parent_sample.join(sample)  # Join means concat in Pandas.


# ----------------------------------------------------------------
#      Sampling from data tables
# ----------------------------------------------------------------


def load_data_table(table, join_keys, usecols):
    return data_utils.load_table(table,
                                 usecols=usecols,
                                 dtype={k: np.int64 for k in join_keys})


class DataTableActor(object):

    def __init__(self, table_name, join_keys, df, join_name):
        self.table = table_name
        self.df = df
        self.join_keys = [f"{table_name}.{k}" for k in join_keys]
        self.df.columns = [f"{table_name}.{k}" for k in self.df.columns]
        self.indicator_column = f"__in_{table_name}"
        self.sample_columns = [
            c for c in self.df.columns if c not in self.join_keys
        ]  # exclude join keys
        self.index_provider = rustlib.IndexProvider(
            f"{join_name}/{table_name}.pk.indices", NULL)
        log.info(f"DataTableActor of `{table_name}` is ready.")

    def ready(self):
        """A remote caller calls this method to check if initialization is done."""
        return True

    def construct_sample(self, join_count_sample):
        join_count_sample = join_count_sample[self.join_keys]
        sample_index = join_count_sample.index
        indices = self.index_provider.sample_indices(join_count_sample.values)
        nonnulls = indices != NULL
        df = self.df.iloc[indices[nonnulls]][self.sample_columns]
        df.set_index(sample_index[nonnulls], inplace=True)
        df[self.indicator_column] = 1
        # Reindexing automatically adds NaN rows
        df = df.reindex(sample_index)
        return df


# ----------------------------------------------------------------
#      Main Sampler
# ----------------------------------------------------------------


def load_jct(table, join_name):
    return data_utils.load(f"{join_name}/{table}.jct",
                           f"join count table of `{table}`")


def _make_sampling_table_ordering(tables, root_name):
    """
    Returns a list of table names with the join_root at the front.
    """
    return [root_name
           ] + [table.name for table in tables if table.name != root_name]


class FactorizedSampler(object):
    """Unbiased join sampler using the Exact Weight algorithm."""

    def __init__(self,
                 loaded_tables,
                 join_spec,
                 sample_batch_size,
                 rng=None,
                 disambiguate_column_names=True,
                 add_full_join_indicators=True,
                 add_full_join_fanouts=True):
        prepare_utils.prepare(join_spec)
        self.join_spec = join_spec
        self.sample_batch_size = sample_batch_size
        self.rng = rng
        self.disambiguate_column_names = disambiguate_column_names
        self.add_full_join_indicators = add_full_join_indicators
        self.add_full_join_fanouts = add_full_join_fanouts
        self.dt_actors = [
            DataTableActor(table.name, join_spec.join_keys[table.name],
                           table.data, join_spec.join_name)
            for table in loaded_tables
        ]
        jcts = {
            table: load_jct(table, join_spec.join_name)
            for table in join_spec.join_tables
        }
        self.jct_actors = {
            table: JoinCountTableActor(table, jct, join_spec)
            for table, jct in jcts.items()
        }
        self.sampling_tables_ordering = _make_sampling_table_ordering(
            loaded_tables, join_spec.join_root)
        self.all_columns = None
        self.rename_dict = None
        self.jct_count_columns = get_jct_count_columns(self.join_spec)
        self.fanout_columns = get_fanout_columns(
            self.join_spec) if add_full_join_fanouts else []

        root = join_spec.join_root
        self.join_card = self.jct_actors[root].jct["{}.weight".format(
            root)].sum()

    def take_jct_sample(self):
        sample = None
        for table in self.sampling_tables_ordering:
            sample = self.jct_actors[table].take_sample(sample,
                                                        self.sample_batch_size,
                                                        self.rng)
        return sample

    def _construct_complete_sample(self, join_count_sample):
        table_samples = [
            table.construct_sample(join_count_sample)
            for table in self.dt_actors
        ]
        if self.add_full_join_fanouts:
            df_cnt = join_count_sample[self.jct_count_columns]
            df_cnt.columns = self.fanout_columns
            table_samples.append(df_cnt)
        ret = pd.concat(table_samples, axis=1)
        return ret

    def _rearrange_columns(self, df):
        """Rearranges the output columns into the conventional order."""
        if self.all_columns is None:
            content_columns = [c for c in df.columns if not c.startswith("_")]
            indicator_columns = [
                "__in_{}".format(t) for t in self.join_spec.join_tables
            ] if self.add_full_join_indicators else []
            fanout_columns = self.fanout_columns
            self.all_columns = content_columns + indicator_columns + fanout_columns
            if self.disambiguate_column_names:
                self.rename_dict = {
                    c: c.replace(".", ":")
                    for c in df.columns
                    if not c.startswith("_")
                }
            else:  # used in make_job_queries.py
                self.rename_dict = {
                    c: ".".join(c.split(".")[-2:])
                    for c in df.columns
                    if not c.startswith("_")
                }
        df = df[self.all_columns]
        df.rename(self.rename_dict, axis=1, inplace=True)
        return df

    def run(self):
        join_count_sample = self.take_jct_sample()
        full_sample = self._construct_complete_sample(join_count_sample)
        full_sample = self._rearrange_columns(full_sample)
        full_sample.replace(NULL, np.nan, inplace=True)
        return full_sample


class FactorizedSamplerIterDataset(common.SamplerBasedIterDataset):
    """An IterableDataset that scales to multiple equivalence classes."""

    def _init_sampler(self):
        self.sampler = FactorizedSampler(self.tables, self.join_spec,
                                         self.sample_batch_size, self.rng,
                                         self.disambiguate_column_names,
                                         self.add_full_join_indicators,
                                         self.add_full_join_fanouts)

    def _run_sampler(self):
        return self.sampler.run()


LoadedTable = collections.namedtuple("LoadedTable", ["name", "data"])


def main():
    config = experiments.JOB_FULL
    join_spec = join_utils.get_join_spec(config)
    prepare_utils.prepare(join_spec)
    loaded_tables = []
    for t in join_spec.join_tables:
        print('Loading', t)
        table = datasets.LoadImdb(t, use_cols=config["use_cols"])
        table.data.info()
        loaded_tables.append(table)

    t_start = time.time()
    join_iter_dataset = FactorizedSamplerIterDataset(
        loaded_tables,
        join_spec,
        sample_batch_size=1000 * 100,
        disambiguate_column_names=True)

    table = common.ConcatTables(loaded_tables,
                                join_spec.join_keys,
                                sample_from_join_dataset=join_iter_dataset)

    join_iter_dataset = common.FactorizedSampleFromJoinIterDataset(
        join_iter_dataset,
        base_table=table,
        factorize_blacklist=[],
        word_size_bits=10,
        factorize_fanouts=True)
    t_end = time.time()
    log.info(f"> Initialization took {t_end - t_start} seconds.")

    join_iter_dataset.join_iter_dataset._sample_batch()
    print('-' * 60)
    print("Done")


if __name__ == "__main__":
    main()
