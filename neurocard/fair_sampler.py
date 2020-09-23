#!/usr/bin/env python3
"""(DEPRECATED) A reference implementation of an unbiased join sampler.

Compared to factorized_sampler, which samples in an online, "factorized"
fashion (walking the join tree), this implementation naively pre-computes the
entire join counts first.  This is less efficient than the proper
factorized_sampler module.

Using this class is not recommended.
"""
import argparse
import collections
import hashlib
import os
import re
import pickle

import glog as log
import networkx as nx
import numpy as np
import pandas as pd
import ray

from common import time_this, JoinTableAndColumnNames, SamplerBasedIterDataset
import datasets
import experiments
import join_utils

PRIMARY_RELATION = "title"
CACHE_DIR = "./cache_fair/"

# ----------------------------------------------------------------
#      Creating the join count table
# ----------------------------------------------------------------


def _get_join_tables_and_columns(clause):
    """Parses a join clause like `<t1>.<c1>=<t2>.<c2>` into tables and columns."""
    pattern = re.compile(r"^(.*)\.(.*?)\s*=\s*(.*)\.(.*)$")
    match = pattern.match(clause)
    assert match, clause
    return match.group(1), match.group(2), match.group(3), match.group(4)


def get_table_info(join_clauses):
    """
    Parses join clauses.

    Returns a dictionary of tables to a list of columns in the table used in the join,
    and a list of 4-tuples of parsed join clauses.
    """
    table_info = collections.defaultdict(set)
    parsed_join_clauses = []
    for clause in join_clauses:
        t1, c1, t2, c2 = _get_join_tables_and_columns(clause)
        parsed_join_clauses.append((t1, c1, t2, c2))
        table_info[t1].add(c1)
        table_info[t2].add(c2)
    return table_info, parsed_join_clauses


def make_join_graph(parsed_join_clauses):
    """Constructs an undirected graph in which vertices are tables, and edges are joins."""
    g = nx.Graph()
    for t1, c1, t2, c2 in parsed_join_clauses:
        g.add_node(t1)
        g.add_node(t2)
        if not g.has_edge(t1, t2):
            g.add_edge(t1, t2, join_columns={t1: [c1], t2: [c2]})
        else:
            edge = g[t1][t2]
            edge["join_columns"][t1].append(c1)
            edge["join_columns"][t2].append(c2)
    return g


@time_this
def make_count_tables(table_info, tables):
    count_tables = {}
    for table_name, join_columns in table_info.items():
        table = tables[table_name]
        join_columns = list(join_columns)
        ct = table.groupby(by=join_columns).size().reset_index(name="cnt")
        if len(join_columns) > 1:
            per_key_count_tables = [
                table.groupby(by=c).size().reset_index(name="{}:cnt".format(c))
                for c in join_columns
            ]
            for k, kct in zip(join_columns, per_key_count_tables):
                ct = ct.merge(kct, how="left", on=k)
        else:
            ct["{}:cnt".format(join_columns[0])] = ct["cnt"]
        count_tables[table_name] = ct
    return count_tables


@time_this
def join_count_tables(table_info, parsed_join_clauses, count_tables, join_how):
    g = make_join_graph(parsed_join_clauses)
    df_ret = None
    for edge in nx.dfs_edges(g, source=parsed_join_clauses[0][0]):
        t1, t2 = edge
        join_columns = g.edges[edge]["join_columns"]
        cs1 = join_columns[t1]  # columns to join
        cs2 = join_columns[t2]  # columns to join
        if df_ret is None:
            df_ret = count_tables[t1].add_prefix(t1 + ":")
        log.info("Joining {} and {} on {}".format(
            t1, t2, ", ".join([c1 + "=" + c2 for c1, c2 in zip(cs1, cs2)])))
        # NOTE: Since we are traversing the join graph in a DFS order, it is
        # guaranteed that at this point `df_ret` already contains `t1.c1`.
        df_ret = df_ret.merge(
            count_tables[t2].add_prefix(t2 + ":"),
            how=join_how,
            left_on=["{}:{}".format(t1, c) for c in cs1],
            right_on=["{}:{}".format(t2, c) for c in cs2],
        )

    # NOTE: `np.nanprod()` treats `np.nan` as 1.
    df_ret["cnt"] = np.nanprod([df_ret[f"{t}:cnt"] for t in table_info], axis=0)
    return df_ret


@time_this
def load_or_create_join_count_table(join_clauses, join_how, filename=None):
    if filename is None:
        filename = ",".join(join_clauses + [join_how])
        max_filename_length = 240
        if len(filename) > max_filename_length:
            h = hashlib.sha1(filename.encode()).hexdigest()[:8]
            filename = filename[:max_filename_length] + "_" + h
        filename += ".df"
    save_path = os.path.join(CACHE_DIR, filename)
    try:
        with open(save_path, "rb") as f:
            log.info("Loading join count table from {}".format(save_path))
            df = pickle.load(f)
        assert isinstance(df, pd.DataFrame), type(df)
        return df
    except:
        log.info("Creating the join count table.")
        table_info, parsed_join_clauses = get_table_info(join_clauses)
        log.info("Loading tables {}".format(", ".join(t for t in table_info)))
        data_tables = {
            table: datasets.LoadImdb(table, use_cols=None).data
            for table, cols in table_info.items()
        }

        log.info("Making count tables")
        count_tables = make_count_tables(table_info, data_tables)

        log.info("Joining count tables")
        df = join_count_tables(table_info, parsed_join_clauses, count_tables,
                               join_how)

        log.info("Calculated full join size = {}".format(df["cnt"].sum()))

        log.info("Saved join count table to {}".format(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(df, f, protocol=4)
        return df


# ----------------------------------------------------------------
#      Fair Sampler
# ----------------------------------------------------------------


def _build_groupby_indices(df, table_name, join_columns):
    """
    Pre-computes indexes based on the group-by columns.
    Returns a dictionary of tuples to the list of indices.
    """
    log.info("Grouping table '{}' by: {}.".format(table_name,
                                                  ", ".join(join_columns)))
    ret = df.groupby(join_columns).indices
    if len(join_columns) == 1:
        # Manually patch the dictionary to make sure its keys are tuples.
        ret = {(k,): v for k, v in ret.items()}
    return ret


def _pick_index_from_groups(groups, tup, rng):
    indices = groups.get(tup)
    if indices is None:
        return np.nan
    i = rng.randint(indices.size)
    return indices[i]


# TODO: Figure out why ray.remote does not work with PyTorch dataloaders.
# @ray.remote
class TableManager(object):

    def __init__(self, table, join_columns):
        table_name = table.name
        log.info("Initializing table {}.".format(table_name))
        self.table_name = table_name
        self.df = table.data

        # Prepend table names to join columns to match the column names
        # in the join count table.
        self.join_columns = [
            JoinTableAndColumnNames(table_name, c, sep=":")
            for c in join_columns
        ]
        self.df.columns = [
            JoinTableAndColumnNames(table_name, c, sep=":")
            for c in self.df.columns
        ]
        self.groups = _build_groupby_indices(self.df, table_name,
                                             self.join_columns)
        self.indicator_column = "__in_{}".format(self.table_name)

    def construct_sample(self, join_count_sample, exclude_join_keys,
                         add_full_join_indicators, rng):
        join_count_sample = join_count_sample[self.join_columns]
        sample_index = join_count_sample.index
        indices = np.array([
            _pick_index_from_groups(self.groups, tup, rng)
            for tup in join_count_sample.itertuples(index=False)
        ])
        nonnulls = ~np.isnan(indices)
        df = self.df.iloc[indices[nonnulls]].set_index(sample_index[nonnulls])
        if add_full_join_indicators:
            df[self.indicator_column] = 1
        if exclude_join_keys:
            df.drop(columns=self.join_columns, inplace=True)
        # Reindexing automatically adds NaN rows
        df = df.reindex(sample_index)
        return df


def get_table_info_from_join_count_table(df):
    ret = collections.defaultdict(list)
    for col in df.columns:
        pieces = col.split(":")
        if len(pieces) != 2:
            continue
        t = pieces[0]
        c = pieces[1]
        if c == "cnt":
            continue
        ret[t].append(c)
    for cs in ret.values():
        cs.sort()
    return ret


def _get_fanout_columns(table_info):
    """
    If a table only has one join key, then use `__fanout_{t}` for backward
    compatibility; otherwise, use `__fanout_{t}__{c}` per key.
    """
    ret = []
    for t, cs in table_info.items():
        if t == PRIMARY_RELATION:
            continue
        if len(cs) == 1:
            ret.append("__fanout_{}".format(t))
        else:
            for c in cs:
                ret.append("__fanout_{}__{}".format(t, c))
    return ret


def construct_complete_sample(join_count_sample, table_info, data_tables,
                              exclude_join_keys, add_full_join_indicators,
                              add_full_join_fanouts, rng):
    # TODO: Use this version when ray.remote works
    # table_samples = [
    #     table.construct_sample.remote(join_count_sample, exclude_join_keys,
    #                                   add_full_join_indicators, rng)
    #     for table in data_tables
    # ]
    # table_samples = ray.get(table_samples)
    table_samples = [
        table.construct_sample(join_count_sample, exclude_join_keys,
                               add_full_join_indicators, rng)
        for table in data_tables
    ]
    if add_full_join_fanouts:
        df_cnt = join_count_sample[[
            "{}:{}:cnt".format(t, c) for t, cs in table_info.items()
            if t != PRIMARY_RELATION for c in cs
        ]]
        df_cnt.columns = _get_fanout_columns(table_info)
        table_samples.append(df_cnt)
    ret = pd.concat(table_samples, axis=1)
    return ret


def _make_distribution(series):
    """Make a probability distribution out of a series of counts."""
    arr = series.values
    return arr / np.sum(arr)


class FairSampler(object):
    """
    join_spec: if join_graph is None, join_keys is used to deduce a
    single equivalence class join graph.
    tables: loaded tables.
    """

    def __init__(self,
                 join_spec,
                 tables,
                 sample_batch_size,
                 disambiguate_column_names=False,
                 exclude_join_keys=True,
                 add_full_join_indicators=True,
                 add_full_join_fanouts=True):
        self.join_count_table = load_or_create_join_count_table(
            join_spec.join_clauses,
            "outer",
            filename="{}.df".format(join_spec.join_name))
        self.jct_distribution = _make_distribution(self.join_count_table["cnt"])
        self.table_info = get_table_info_from_join_count_table(
            self.join_count_table)
        self.table_names = join_spec.join_tables
        log.info("Total rows: {}.".format(len(self.join_count_table.index)))
        log.info("Total join size: {}.".format(
            self.join_count_table["cnt"].sum()))
        log.info("Joined tables: {}.".format(self.table_names))
        expected_card = datasets.JoinOrderBenchmark.GetFullOuterCardinalityOrFail(
            self.table_names)
        assert expected_card == self.join_count_table["cnt"].sum(), (
            expected_card, self.join_count_table["cnt"].sum())

        self.data_tables = [
            TableManager(table, self.table_info[table.name])
            # TODO: Use when ray.remote works
            # TableManager.remote(table, self.table_info[table])
            for table in tables
        ]

        self.sample_batch_size = sample_batch_size
        self.disambiguate_column_names = disambiguate_column_names
        self.exclude_join_keys = exclude_join_keys
        self.add_full_join_indicators = add_full_join_indicators
        self.add_full_join_fanouts = add_full_join_fanouts
        self.all_columns = None

    def _rearrange_columns(self, df):
        """Rearranges the output columns into the conventional order."""
        if self.all_columns is None:
            content_columns = [c for c in df.columns if not c.startswith("_")]
            indicator_columns = ["__in_{}".format(t) for t in self.table_names
                                ] if self.add_full_join_indicators else []
            fanout_columns = _get_fanout_columns(
                self.table_info) if self.add_full_join_fanouts else []
            self.all_columns = content_columns + indicator_columns + fanout_columns
        df = df[self.all_columns]
        if not self.disambiguate_column_names:
            df.columns = [
                c if c.startswith("_") else c.split(":")[1] for c in df.columns
            ]
        return df

    def run(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        idx = rng.choice(self.join_count_table.index,
                         size=self.sample_batch_size,
                         replace=True,
                         p=self.jct_distribution)
        join_count_sample = self.join_count_table.loc[idx].reset_index(
            drop=True)
        full_sample = construct_complete_sample(join_count_sample,
                                                self.table_info,
                                                self.data_tables,
                                                self.exclude_join_keys,
                                                self.add_full_join_indicators,
                                                self.add_full_join_fanouts, rng)
        full_sample = self._rearrange_columns(full_sample)
        return full_sample


class FairSamplerIterDataset(SamplerBasedIterDataset):
    """An IterableDataset that supports multiple equivalence classes."""

    def _init_sampler(self):
        self.fair_sampler = FairSampler(self.join_spec, self.tables,
                                        self.sample_batch_size,
                                        self.disambiguate_column_names,
                                        self.add_full_join_indicators,
                                        self.add_full_join_fanouts)

    def _run_sampler(self):
        return self.fair_sampler.run(rng=self.rng)


# ----------------------------------------------------------------
#      Testing/Benchmarking
# ----------------------------------------------------------------


def test_job_light():
    print("==== Single equivalence class example (JOB-Light) ====")
    config = experiments.JOB_LIGHT_BASE
    join_spec = join_utils.get_join_spec(config)
    test_sampler(join_spec)


def test_job_m():
    print("==== JOB-M example ====")
    config = experiments.JOB_M
    join_spec = join_utils.get_join_spec(config)
    test_sampler(join_spec)


def test_sampler(join_spec, batch_size=1024 * 16):
    tables = [
        datasets.LoadImdb(t, use_cols="multi") for t in join_spec.join_tables
    ]
    sampler = FairSampler(join_spec, tables, batch_size)
    print("-" * 60)
    print("initialization done")
    print("-" * 60)

    sample = time_this(sampler.run)()
    print(sample)
    print(sample.columns)


def main():
    ray.init(ignore_reinit_error=True)
    log.setLevel("DEBUG")

    test_job_light()
    test_job_m()


if __name__ == "__main__":
    main()
