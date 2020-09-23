"""Generates new queries on the JOB-light schema.

For each JOB-light join template, repeat #queries per template:
   - sample a tuple from this inner join result via factorized_sampler
   - sample #filters, and columns to put these filters on
   - query literals: use this tuple's values
   - sample ops: {>=, <=, =} for numerical columns and = for categoricals.

Uses either Spark or Postgres for actual execution to obtain true
cardinalities.  Both setups require data already be loaded.  For Postgres, the
server needs to be live.

Typical usage:

To generate queries:
    python make_job_queries.py --output_csv <csv> --num_queries <n>

To print selectivities of already generated queries:
    python make_job_queries.py \
      --print_sel --output_csv queries/job-light.csv --num_queries 70
"""

import os
import subprocess
import textwrap
import time

from absl import app
from absl import flags
from absl import logging
from mako import template
import numpy as np
import pandas as pd
import psycopg2 as pg
from pyspark.sql import SparkSession

import common
import datasets
from factorized_sampler import FactorizedSamplerIterDataset
import join_utils
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'job_light_csv', 'queries/job-light.csv',
    'Path to the original JOB-light queries; used for loading templates.')
flags.DEFINE_integer(
    'num_queries', 1000,
    'Number of queries to generate, distributed across templates.')
flags.DEFINE_integer('min_filters', 3,
                     'Minimum # of filters (inclusive) a query can have.')
flags.DEFINE_integer('max_filters', 7,
                     'Maximum # of filters (inclusive) a query can have.')
flags.DEFINE_string(
    'db', 'dbname=imdb host=127.0.0.1',
    'Configs for connecting to Postgres server (must be launched and running).')
flags.DEFINE_string('output_csv', 'job-light-range-spark.csv',
                    'Path to CSV file to output into.')

# Spark configs.
flags.DEFINE_string('spark_master', 'local[*]', 'spark.master.')

# Print selectivities.
flags.DEFINE_bool(
    'print_sel', False, 'If specified, load generated cardinalities from '
    '--output_csv and print selectivities.')

JOB_LIGHT_OUTER_CARDINALITY = 2128877229383


def ExecuteSql(spark, sql):
    df = spark.sql(sql.replace(';', ''))
    return df.collect()


def MakeQueries(spark, cursor, num_queries, tables_in_templates, table_names,
                join_keys, rng):
    """Sample a tuple from actual join result then place filters."""
    spark.catalog.clearCache()

    # TODO: this assumes single equiv class.
    join_items = list(join_keys.items())
    lhs = join_items[0][1]
    join_clauses_list = []
    for rhs in join_items[1:]:
        rhs = rhs[1]
        join_clauses_list.append('{} = {}'.format(lhs, rhs))
        lhs = rhs
    join_clauses = '\n AND '.join(join_clauses_list)

    # Take only the content columns.
    content_cols = []
    categoricals = []
    numericals = []
    for table_name in table_names:
        categorical_cols = datasets.JoinOrderBenchmark.CATEGORICAL_COLUMNS[
            table_name]
        for c in categorical_cols:
            disambiguated_name = common.JoinTableAndColumnNames(table_name,
                                                                c,
                                                                sep='.')
            content_cols.append(disambiguated_name)
            categoricals.append(disambiguated_name)

        range_cols = datasets.JoinOrderBenchmark.RANGE_COLUMNS[table_name]
        for c in range_cols:
            disambiguated_name = common.JoinTableAndColumnNames(table_name,
                                                                c,
                                                                sep='.')
            content_cols.append(disambiguated_name)
            numericals.append(disambiguated_name)

    # Build a concat table representing the join result schema.
    join_keys_list = [join_keys[n] for n in table_names]
    join_spec = join_utils.get_join_spec({
        "join_tables": table_names,
        "join_keys": dict(
            zip(table_names, [[k.split(".")[1]] for k in join_keys_list])),
        "join_root": "title",
        "join_how": "inner",
    })
    ds = FactorizedSamplerIterDataset(tables_in_templates,
                                      join_spec,
                                      sample_batch_size=num_queries,
                                      disambiguate_column_names=False,
                                      add_full_join_indicators=False,
                                      add_full_join_fanouts=False)
    concat_table = common.ConcatTables(tables_in_templates,
                                       join_keys_list,
                                       sample_from_join_dataset=ds)

    template_for_execution = template.Template(
        textwrap.dedent("""
        SELECT COUNT(*)
        FROM ${', '.join(table_names)}
        WHERE ${join_clauses}
        AND ${filter_clauses};
    """).strip())

    true_inner_join_card = ds.sampler.join_card
    true_full_join_card = JOB_LIGHT_OUTER_CARDINALITY
    print('True inner join card', true_inner_join_card, 'true full',
          true_full_join_card)

    ncols = len(content_cols)
    queries = []
    filter_strings = []
    sql_queries = []  # To get true cardinalities.

    while len(queries) < num_queries:
        sampled_df = ds.sampler.run()[content_cols]

        for r in sampled_df.iterrows():
            tup = r[1]
            num_filters = rng.randint(FLAGS.min_filters,
                                      max(ncols // 2, FLAGS.max_filters))

            # Positions where the values are non-null.
            non_null_indices = np.argwhere(~pd.isnull(tup).values).reshape(-1,)
            if len(non_null_indices) < num_filters:
                continue
            print('{} filters out of {} content cols'.format(
                num_filters, ncols))

            # Place {'<=', '>=', '='} on numericals and '=' on categoricals.
            idxs = rng.choice(non_null_indices, replace=False, size=num_filters)
            vals = tup[idxs].values
            cols = np.take(content_cols, idxs)
            ops = rng.choice(['<=', '>=', '='], size=num_filters)
            sensible_to_do_range = [c in numericals for c in cols]
            ops = np.where(sensible_to_do_range, ops, '=')

            print('cols', cols, 'ops', ops, 'vals', vals)

            queries.append((cols, ops, vals))
            filter_strings.append(','.join(
                [','.join((c, o, str(v))) for c, o, v in zip(cols, ops, vals)]))

            # Quote string literals & leave other literals alone.
            filter_clauses = '\n AND '.join([
                '{} {} {}'.format(col, op, val)
                if concat_table[col].data.dtype in [np.int64, np.float64] else
                '{} {} \'{}\''.format(col, op, val)
                for col, op, val in zip(cols, ops, vals)
            ])

            sql = template_for_execution.render(table_names=table_names,
                                                join_clauses=join_clauses,
                                                filter_clauses=filter_clauses)
            sql_queries.append(sql)

            if len(queries) >= num_queries:
                break

    true_cards = []

    for i, sql_query in enumerate(sql_queries):
        DropBufferCache()
        spark.catalog.clearCache()

        print('  Query',
              i,
              'out of',
              len(sql_queries),
              '[{}]'.format(filter_strings[i]),
              end='')

        t1 = time.time()

        true_card = ExecuteSql(spark, sql_query)[0][0]

        # cursor.execute(sql_query)
        # result = cursor.fetchall()
        # true_card = result[0][0]

        dur = time.time() - t1

        true_cards.append(true_card)
        print(
            '...done: {} (inner join sel {}; full sel {}; inner join {}); dur {:.1f}s'
            .format(true_card, true_card / true_inner_join_card,
                    true_card / true_full_join_card, true_inner_join_card, dur))

        # if i > 0 and i % 1 == 0:
        #     spark = StartSpark(spark)

    df = pd.DataFrame({
        'tables': [','.join(table_names)] * len(true_cards),
        'join_conds': [
            ','.join(map(lambda s: s.replace(' ', ''), join_clauses_list))
        ] * len(true_cards),
        'filters': filter_strings,
        'true_cards': true_cards,
    })
    df.to_csv(FLAGS.output_csv, sep='#', mode='a', index=False, header=False)
    print('Template done.')
    return queries, true_cards


def StartSpark(spark=None):
    spark = SparkSession.builder.appName('make_job_queries')\
        .config('spark.master', FLAGS.spark_master)\
        .config('spark.driver.memory', '200g')\
        .config('spark.eventLog.enabled', 'true')\
        .config('spark.sql.warehouse.dir', '/home/ubuntu/spark-sql-warehouse')\
        .config('spark.sql.cbo.enabled', 'true')\
        .config('spark.memory.fraction', '0.9')\
        .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')\
        .config('spark.memory.offHeap.enabled', 'true')\
        .config('spark.memory.offHeap.size', '100g')\
        .enableHiveSupport()\
        .getOrCreate()

    print('Launched spark:', spark.sparkContext)
    executors = str(
        spark.sparkContext._jsc.sc().getExecutorMemoryStatus().keys().mkString(
            '\n ')).strip()
    print('{} executors:\n'.format(executors.count('\n') + 1), executors)
    return spark


def DropBufferCache():
    worker_addresses = os.path.expanduser('~/hosts-workers')
    if os.path.exists(worker_addresses):
        # If distributed, drop each worker.
        print(
            str(
                subprocess.check_output([
                    'parallel-ssh', '-h', worker_addresses, '--inline-stdout',
                    'sync && sudo bash -c  \'echo 3 > /proc/sys/vm/drop_caches\' && free -h'
                ])))
    else:
        # Drop this machine only.
        subprocess.check_output(['sync'])
        subprocess.check_output(
            ['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'])
        subprocess.check_output(['free', '-h'])


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def MakeTablesKey(table_names):
    sorted_tables = sorted(table_names)
    return '-'.join(sorted_tables)


def main(argv):
    del argv  # Unused.

    # conn = pg.connect(FLAGS.db)
    # conn.set_session(autocommit=True)
    # cursor = conn.cursor()
    cursor = None

    tables = datasets.LoadImdb(use_cols=None)

    # Load all templates in original JOB-light.
    queries = utils.JobToQuery(FLAGS.job_light_csv, use_alias_keys=False)
    tables_to_join_keys = {}
    for query in queries:
        key = MakeTablesKey(query[0])
        if key not in tables_to_join_keys:
            join_dict = query[1]
            # Disambiguate: title->id changed to title->title.id.
            for table_name in join_dict.keys():
                # TODO: only support a single join key
                join_key = next(iter(join_dict[table_name]))
                join_dict[table_name] = common.JoinTableAndColumnNames(
                    table_name, join_key, sep='.')
            tables_to_join_keys[key] = join_dict

    num_templates = len(tables_to_join_keys)
    num_queries_per_template = FLAGS.num_queries // num_templates
    logging.info('%d join templates', num_templates)

    rng = np.random.RandomState(1234)
    queries = []  # [(cols, ops, vals)]

    # Disambiguate to not prune away stuff during join sampling.
    for table_name, table in tables.items():
        for col in table.columns:
            col.name = common.JoinTableAndColumnNames(table.name,
                                                      col.name,
                                                      sep='.')
        table.data.columns = [col.name for col in table.columns]

    if FLAGS.print_sel:
        # Print true selectivities.
        df = pd.read_csv(FLAGS.output_csv, sep='#', header=None)
        assert len(df) == FLAGS.num_queries, (len(df), FLAGS.num_queries)

        inner = []
        true_inner_card_cache = {}

        for row in df.iterrows():
            vs = row[1]
            table_names, join_clauses, true_card = vs[0], vs[1], vs[3]
            table_names = table_names.split(',')
            print('Template: {}\tTrue card: {}'.format(table_names, true_card))

            # JOB-light: contains 'full_name alias'.
            # JOB-light-ranges: just 'full_name'.
            if ' ' in table_names[0]:
                table_names = [n.split(' ')[0] for n in table_names]

            tables_in_templates = [tables[n] for n in table_names]
            key = MakeTablesKey(table_names)
            join_keys_list = tables_to_join_keys[key]

            if key not in true_inner_card_cache:
                join_spec = join_utils.get_join_spec({
                    "join_tables": table_names,
                    "join_keys": dict(
                        zip(table_names,
                            [[k.split(".")[1]] for k in join_keys_list])),
                    "join_root": "title",
                    "join_how": "inner",
                })
                ds = FactorizedSamplerIterDataset(
                    tables_in_templates,
                    join_spec,
                    sample_batch_size=num_queries,
                    disambiguate_column_names=False,
                    add_full_join_indicators=False,
                    add_full_join_fanouts=False)
                true_inner_card_cache[key] = ds.sampler.join_card
            inner.append(true_inner_card_cache[key])

        pd.DataFrame({
            'true_cards': df[3],
            'true_inner': inner,
            'inner_sel': df[3] * 1.0 / inner,
            'outer_sel': df[3] * 1.0 / JOB_LIGHT_OUTER_CARDINALITY
        }).to_csv(FLAGS.output_csv + '.sel', index=False)
        print('Done:', FLAGS.output_csv + '.sel')

    else:
        # Generate queries.
        last_run_queries = file_len(FLAGS.output_csv) if os.path.exists(
            FLAGS.output_csv) else 0
        next_template_idx = last_run_queries // num_queries_per_template
        print('next_template_idx', next_template_idx)
        print(tables_to_join_keys.items())

        spark = StartSpark()
        for i, (tables_to_join,
                join_keys) in enumerate(tables_to_join_keys.items()):

            if i < next_template_idx:
                print('Skipping template:', tables_to_join)
                continue
            print('Template:', tables_to_join)

            if i == num_templates - 1:
                num_queries_per_template += FLAGS.num_queries % num_templates

            # Generate num_queries_per_template.
            table_names = tables_to_join.split('-')

            tables_in_templates = [tables[n] for n in table_names]

            queries.extend(
                MakeQueries(spark, cursor, num_queries_per_template,
                            tables_in_templates, table_names, join_keys, rng))


if __name__ == '__main__':
    app.run(main)
