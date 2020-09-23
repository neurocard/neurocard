"""Utility functions."""

import ast
from collections import defaultdict
import csv


def _get_table_dict(tables):
    table_dict = {}
    for t in tables:
        split = t.split(' ')
        if len(split) > 1:
            # Alias -> full table name.
            table_dict[split[1]] = split[0]
        else:
            # Just full table name.
            table_dict[split[0]] = split[0]
    return table_dict


def _get_join_dict(joins, table_dict, use_alias_keys):
    join_dict = defaultdict(set)
    for j in joins:
        ops = j.split('=')
        op1 = ops[0].split('.')
        op2 = ops[1].split('.')
        t1, k1 = op1[0], op1[1]
        t2, k2 = op2[0], op2[1]
        if not use_alias_keys:
            t1 = table_dict[t1]
            t2 = table_dict[t2]
        join_dict[t1].add(k1)
        join_dict[t2].add(k2)
    return join_dict


def _try_parse_literal(s):
    try:
        ret = ast.literal_eval(s)
        # IN needs a tuple operand
        # String equality needs a string operand
        if isinstance(ret, tuple) or isinstance(ret, str):
            return ret
        return s
    except:
        return s


def _get_predicate_dict(predicates, table_dict):
    predicates = [predicates[x:x + 3] for x in range(0, len(predicates), 3)]
    predicate_dict = {}
    for p in predicates:
        split_p = p[0].split('.')
        table_name = table_dict[split_p[0]]
        if table_name not in predicate_dict:
            predicate_dict[table_name] = {}
            predicate_dict[table_name]['cols'] = []
            predicate_dict[table_name]['ops'] = []
            predicate_dict[table_name]['vals'] = []
        predicate_dict[table_name]['cols'].append(split_p[1])
        predicate_dict[table_name]['ops'].append(p[1])
        predicate_dict[table_name]['vals'].append(_try_parse_literal(p[2]))
    return predicate_dict


def JobToQuery(csv_file, use_alias_keys=True):
    """Parses custom #-delimited query csv.

    `use_alias_keys` only applies to the 2nd return value.
    If use_alias_keys is true, join_dict will use aliases (t, mi) as keys;
    otherwise it uses real table names (title, movie_index).

    Converts into (tables, join dict, predicate dict, true cardinality).  Only
    works for single equivalency class.
    """
    queries = []
    with open(csv_file) as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            reader = csv.reader(row)  # comma-separated
            table_dict = _get_table_dict(next(reader))
            join_dict = _get_join_dict(next(reader), table_dict, use_alias_keys)
            predicate_dict = _get_predicate_dict(next(reader), table_dict)
            true_cardinality = int(next(reader)[0])
            queries.append((list(table_dict.values()), join_dict,
                            predicate_dict, true_cardinality))

        return queries


def UnpackQueries(concat_table, queries):
    """Converts custom query representation to (cols, ops, vals)."""
    converted = []
    true_cards = []
    for q in queries:
        tables, join_dict, predicate_dict, true_cardinality = q
        # All predicates in a query (an AND of these).
        query_cols, query_ops, query_vals = [], [], []

        skip = False
        # A naive impl of "is join graph subset of another join" check.
        for table in tables:
            if table not in concat_table.table_names:
                print('skipping query')
                skip = True
                break
            # Add the indicators.
            idx = concat_table.ColumnIndex('__in_{}'.format(table))
            query_cols.append(concat_table.columns[idx])
            query_ops.append('=')
            query_vals.append(1)

        if skip:
            continue

        for table, preds in predicate_dict.items():
            cols = preds['cols']
            ops = preds['ops']
            vals = preds['vals']
            assert len(cols) == len(ops) and len(ops) == len(vals)
            for c, o, v in zip(cols, ops, vals):
                column = concat_table.columns[concat_table.TableColumnIndex(
                    table, c)]
                query_cols.append(column)
                query_ops.append(o)
                # Cast v into the correct column dtype.
                cast_fn = column.all_distinct_values.dtype.type
                # If v is a collection, cast its elements.
                if isinstance(v, (list, set, tuple)):
                    qv = type(v)(map(cast_fn, v))
                else:
                    qv = cast_fn(v)
                query_vals.append(qv)

        converted.append((query_cols, query_ops, query_vals))
        true_cards.append(true_cardinality)
    return converted, true_cards


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def HumanFormat(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
