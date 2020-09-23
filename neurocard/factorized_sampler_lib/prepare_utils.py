#!/usr/bin/env python3

import argparse
import os
import re

import glog as log
import numpy as np
import pandas as pd
import ray

from factorized_sampler_lib import data_utils
from factorized_sampler_lib import rustlib
import join_utils

NULL = -1


@ray.remote
def get_first_jct(join_name, table, base_count_table):

    @data_utils.save_result(f"{table}.jct", join_name,
                            f"join count table of `{table}`")
    def work(table, base_count_table):
        log.info(f"Creating join count table for `{table}`.")
        ret = base_count_table
        ret.columns = [f"{table}.{k}" for k in ret.columns]
        return ret

    return work(table, base_count_table)


@ray.remote
def get_jct(table, base_count_table, dependencies, dependency_jcts, join_spec):

    @data_utils.save_result(f"{table}.jct", join_spec.join_name,
                            f"join count table of `{table}`")
    def work(table, bct, dependencies, dependency_jcts, join_spec):
        """
        The base count table (BCT) contains the following columns:
          {k1}, {k2}, ..., weight, {k1}.cnt, {k2}.cnt, ...

        The join count table (JCT) contains the following columns:
          {table}.{k1}, ..., {table}.weight, {table}.{k1}.cnt, ...

        The only difference is that the `weight` values in the JCT are
        aggregated from the dependency JCTs. The fanout counts are copied
        from the BCT. The JCT contains at most one extra row than the BCT,
        namely the NULL row, if the dependency JCTs contain values not in
        the BCT.
        """
        log.info(
            f"Creating join count table for `{table}` from dependencies {dependencies}"
        )
        jct_columns = [f"{table}.{k}" for k in bct.columns]
        bct.columns = jct_columns
        keys = join_spec.join_keys[table]
        groupby_keys = [f"{table}.{k}" for k in keys]
        table_weight = f"{table}.weight"
        ret_keys = groupby_keys + [table_weight]
        ret = bct[ret_keys]
        dependency_jcts = ray.get(dependency_jcts)
        for other, other_jct in zip(dependencies, dependency_jcts):
            join_keys = join_spec.join_graph[table][other]["join_keys"]
            table_key = f"{table}.{join_keys[table]}"
            other_key = f"{other}.{join_keys[other]}"
            other_weight = f"{other}.weight"
            ret = ret.merge(
                other_jct[[other_key, other_weight]],
                how=join_spec.join_how,
                left_on=table_key,
                right_on=other_key,
            )
            ret[table_weight] = np.nanprod(
                [ret[table_weight], ret[other_weight]], axis=0)
            ret = ret[ret_keys]
            ret = ret.fillna(NULL).groupby(groupby_keys).sum().reset_index()

        # At this point ret contains the aggregated weights. We now need to
        # copy the *.cnt columns from the BCT, and handle the potential NULL
        # row. If a NULL row exists, then `ret` contains one more row than
        # `bct`. Otherwise, they are of the same length.
        #
        # Do not assert this for inner joins.
        if join_spec.join_how == "outer":
            assert 0 <= len(ret) - len(bct) <= 1, (ret, bct)

        # First, we get the BCT minus the weight column, i.e. only the keys and
        # their fanouts. Then we need to concatenate the fanout columns to ret.
        bct_sans_weight = bct.drop(table_weight, axis=1)
        ret = ret.merge(bct_sans_weight,
                        how="left",
                        left_on=groupby_keys,
                        right_on=groupby_keys)
        ret = ret[jct_columns]

        # This fillna(1) sets the fanout columns in the NULL row to be 1. It is
        # a no-op if ret does not contain NULL.
        jct = ret.fillna(1).astype(np.int64, copy=False)
        return jct

    return work(table, base_count_table, dependencies, dependency_jcts,
                join_spec)


def get_join_count_tables(join_spec):
    base_count_tables_dict = {
        table: get_base_count_table.remote(join_spec.join_name, table, keys)
        for table, keys in join_spec.join_keys.items()
    }
    join_count_tables_dict = {}
    # FIXME: properly traverse the tree via bottom-up order.
    for table in join_utils.get_bottom_up_table_ordering(join_spec):
        dependencies = list(join_spec.join_tree.neighbors(table))
        if len(dependencies) == 0:
            jct = get_first_jct.remote(join_spec.join_name, table,
                                       base_count_tables_dict[table])
        else:
            bct = base_count_tables_dict[table]
            dependency_jcts = [join_count_tables_dict[d] for d in dependencies]
            jct = get_jct.remote(table, bct, dependencies, dependency_jcts,
                                 join_spec)
        join_count_tables_dict[table] = jct
    return join_count_tables_dict


@ray.remote
def get_base_count_table(join_name, table, keys):

    @data_utils.save_result(f"{table}.bct", join_name,
                            f"base count table of `{table}`")
    def work(table, keys):
        df = data_utils.load_table(table,
                                   usecols=keys,
                                   dtype={k: pd.Int64Dtype() for k in keys})
        groupby_ss = df.groupby(keys).size()
        bct = groupby_ss.to_frame(name="weight").reset_index()
        for key in keys:
            kct = df.groupby(key).size().rename(f"{key}.cnt")
            bct = bct.merge(kct, how="left", left_on=key, right_index=True)
        return bct.astype(np.int64, copy=False)

    return work(table, keys)


def get_null_set(my_jct, my_key, parent_jct, parent_key):
    parent_keyset = parent_jct[parent_key].unique()
    parent_keyset = parent_keyset[parent_keyset != NULL]
    my_keyset = my_jct[my_key]
    assert my_keyset.dtype == parent_keyset.dtype, (my_keyset.dtype,
                                                    parent_keyset.dtype)
    null_set = my_keyset[~np.isin(my_keyset.values, parent_keyset)]
    return null_set


@ray.remote
def get_join_key_groups(table, jcts, join_spec):
    jct = ray.get(jcts[table])
    parents = list(join_spec.join_tree.predecessors(table))
    if len(parents) == 0:
        return "Skipped"
    parent = parents[0]
    parent_jct = ray.get(jcts[parent])
    join_keys = join_spec.join_graph[parent][table]["join_keys"]
    my_key = f"{table}.{join_keys[table]}"
    parent_key = f"{parent}.{join_keys[parent]}"

    indices = jct.groupby(my_key).indices
    null_set = get_null_set(jct, my_key, parent_jct, parent_key)
    if null_set.size > 0:
        indices[NULL] = null_set.index.values
    indices = {(k,): v for k, v in indices.items()}
    weights = jct[f"{table}.weight"].values
    rustlib.prepare_indices(f"{join_spec.join_name}/{table}.jk.indices",
                            indices, weights)
    return "OK"


@ray.remote
def get_primary_key_groups(table, keys, df, join_spec):
    indices = df.groupby(keys).indices
    # Manually patch the dictionary to make sure its keys are tuples.
    if len(keys) == 1:
        indices = {(k,): v for k, v in indices.items()}
    rustlib.prepare_indices(f"{join_spec.join_name}/{table}.pk.indices",
                            indices, None)
    return "OK"


@ray.remote
def load_data_table(table, join_keys):
    return data_utils.load_table(table,
                                 dtype={k: pd.Int64Dtype() for k in join_keys})


def check_required_files(join_spec):
    for table in join_spec.join_tables:
        for f in [f"{table}.jct", f"{table}.pk.indices"]:
            path = os.path.join(data_utils.CACHE_DIR, join_spec.join_name, f)
            if not os.path.exists(path):
                return False
    return True


def prepare(join_spec):
    """Prepares all required files for the factorized sampler.
    
    The *.bct and *.jct files are in Feather format and can be loaded with
    pd.read_feather(). The *.indices files are in RON format and can be
    loaded in Rust.

    - {table}.bct: For every table, this is the tuple counts grouped by all
      the join keys used in the join spec. i.e.

        SELECT COUNT(*) FROM {table} GROUP BY {join keys}
    
      This is only used to produce the factorized join count tables (*.jct).

    - {table}.jct: For every table, this is its factorized join count table.
      The most important column is `{title}.weight`, which induces the sampling
      probability of a given tuple. This table also contains the fanout
      counts for each key tuple.

    - {table}.jk.indices: This is a reverse lookup table into the join count
      tables. When a row is sampled from the parent JCT, the sampler needs to
      pick a row from the rows in the current JCT that match the parent join
      key. This file is for this purpose: it is a hash map from the parent keys
      to the row IDs in this JCT that match that key.

      This is a weighted distribution because each row has its own join count
      and should be sampled proportionately.
      
    - {table}.pk.indices: The factorized sampler only produces samples of the
      join key columns. To fetch the data columns, one needs to pick a row
      from the original table that matches the join keys in this join sample.
      This file is for this purpose: it is a hash map from the join keys to
      the primary key IDs in the data table that match the keys.

      This is a uniform distribution because each row in the data table should
      be equally likely to be sampled.
    """
    ray.init(ignore_reinit_error=True)
    if check_required_files(join_spec):
        return

    jcts = get_join_count_tables(join_spec)
    dts = {
        table: load_data_table.remote(table, keys)
        for table, keys in join_spec.join_keys.items()
    }
    jk_groups_weights = {
        table: get_join_key_groups.remote(table, jcts, join_spec)
        for table, jct in jcts.items()
    }
    pk_groups = {
        table: get_primary_key_groups.remote(table, keys, dts[table], join_spec)
        for table, keys in join_spec.join_keys.items()
    }

    for table, jkg in jk_groups_weights.items():
        print(table, ray.get(jkg))
    for table, pkg in pk_groups.items():
        print(table, ray.get(pkg))
