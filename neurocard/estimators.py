"""A suite of cardinality estimators.

In parcticular,
  + ProgressiveSampling: inference algorithms for autoregressive density
    estimators; fanout scaling; draws tuples from trained models subject to
    filters.
  + FactorizedProgressiveSampling: subclass that additionally handles column
    factorization.
"""
import collections
import functools
import json
import random
import re
import shutil
import time
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch

import common
import datasets
import distributions
import factorized_sampler
import join_utils
import made
import make_job_queries as query_lib
import transformer
import utils

# Pass VERBOSE=1 to make query evaluation more verbose.
# Set to 2 for even more outputs.
VERBOSE = 'VERBOSE' in os.environ


def profile(func):  # No-op.
    return func


def operator_isnull(xs, _unused_val, negate=False):
    ret = pd.isnull(xs)
    return ~ret if negate else ret


def operator_like(xs, like_pattern, negate=False):
    escaped = re.escape(like_pattern)
    # Check for both cases for pre-v3.7 compatibility.
    if '\\%' in escaped:
        pattern = escaped.replace('\\%', '.*')
    elif '%' in escaped:
        pattern = escaped.replace('%', '.*')
    else:
        pattern = escape
    vfunc = np.vectorize(lambda x: False if pd.isnull(x) else re.match(
        pattern, x) is not None)
    ret = vfunc(xs)
    return ~ret if negate else ret


def operator_true(xs, v):
    return np.ones_like(xs, dtype=np.bool)


def operator_false(xs, v):
    return np.zeros_like(xs, dtype=np.bool)


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '!=': np.not_equal,
    'IN': np.isin,  # Second operand must be a list or an array.
    'LIKE': operator_like,
    'NOT_LIKE': functools.partial(operator_like, negate=True),
    'IS_NULL': operator_isnull,
    'IS_NOT_NULL': functools.partial(operator_isnull, negate=True),
    'ALL_TRUE': operator_true,
    'ALL_FALSE': operator_false,
}


def GetTablesInQuery(columns, values):
    """Returns a set of table names that are joined in a query."""
    q_tables = set()
    for col, val in zip(columns, values):
        if col.name.startswith('__in_') and val == [1]:
            q_tables.add(col.name[len('__in_'):])
    return q_tables


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        if est.name == 'CardEst':
            est.name = str(est)
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))


def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s


def FillInUnqueriedColumns(table, columns, operators, vals):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)

        if os[idx] is None:
            os[idx] = [o]
            vs[idx] = [v]
        else:
            # Multiple clauses on same attribute.
            os[idx].append(o)
            vs[idx].append(v)

    return cs, os, vs


def ConvertLikeToIn(fact_table, columns, operators, vals):
    """Pre-processes a query by converting LIKE predicates to IN predicates.

    Columns refers to the original columns of the table.
    """
    fact_columns = fact_table.columns
    fact_column_names = [fc.name for fc in fact_columns]
    assert len(columns) == len(operators) == len(vals)
    for i in range(len(columns)):
        col, op, val = columns[i], operators[i], vals[i]
        # We don't convert if this column isn't factorized.
        # If original column name isn't in the factorized col names,
        # then this column is factorized.
        # This seems sorta hacky though.
        if op is not None and col.name not in fact_column_names:
            assert len(op) == len(val)
            for j in range(len(op)):
                o, v = op[j], val[j]
                if 'LIKE' in o:
                    new_o = 'IN'
                    valid = OPS[o](col.all_distinct_values, v)
                    new_val = tuple(col.all_distinct_values[valid])
                    op[j] = new_o
                    val[j] = new_val

    assert len(columns) == len(operators) == len(vals)
    return columns, operators, vals


def ProjectQuery(fact_table, columns, operators, vals):
    """Projects these cols, ops, and vals to factorized version.

    Returns cols, ops, vals, dominant_ops where all lists are size
    len(fact_table.columns), in the factorized table's natural column order.

    Dominant_ops is None for all operators that don't use masking.
    <, >, <=, >=, IN all use masking.
    """
    columns, operators, vals = ConvertLikeToIn(fact_table, columns, operators,
                                               vals)
    nfact_cols = len(fact_table.columns)
    fact_cols = []
    fact_ops = []
    fact_vals = []
    fact_dominant_ops = []

    # i is the index of the current factorized column.
    # j is the index of the actual table column that col_i corresponds to.
    j = -1
    for i in range(nfact_cols):
        col = fact_table.columns[i]
        if col.factor_id in [None, 0]:
            j += 1
        op = operators[j]
        fact_cols.append(col)
        if op is None:
            fact_ops.append(None)
            fact_vals.append(None)
            fact_dominant_ops.append(None)
        else:
            val = vals[j]
            fact_ops.append([])
            fact_vals.append([])
            fact_dominant_ops.append([])
            for o, v in zip(op, val):
                if col.factor_id is None:
                    # This column is not factorized.
                    fact_ops[i].append(o)
                    fact_vals[i].append(v)
                    fact_dominant_ops[i] = None
                else:
                    # This column is factorized.  For IN queries, we need to
                    # map the projection overall elements in the tuple.
                    if o == 'IN':
                        if len(v) == 0:
                            fact_ops[i].append('ALL_FALSE')
                            fact_vals[i].append(None)
                            fact_dominant_ops[i].append(None)
                        else:
                            fact_ops[i].append(o)
                            val_list = np.array(list(v))
                            val_list = common.Discretize(
                                columns[j], val_list, fail_out_of_domain=False)
                            assert len(val_list) > 0, val_list
                            p_v_list = np.vectorize(col.ProjectValue)(val_list)
                            fact_vals[i].append(tuple(p_v_list))
                            fact_dominant_ops[i].append('IN')
                    # IS_NULL/IS_NOT_NULL Handling.
                    # IS_NULL+column has null value -> convert to = 0.
                    # IS_NULL+column has no null value -> return False for
                    #   everything.
                    # IS_NOT_NULL+column has null value -> convert to > 0.
                    # IS_NOT_NULL+column has no null value -> return True for
                    #   everything.
                    elif 'NULL' in o:
                        if np.any(pd.isnull(columns[j].all_distinct_values)):
                            if o == 'IS_NULL':
                                fact_ops[i].append(col.ProjectOperator('='))
                                fact_vals[i].append(col.ProjectValue(0))
                                fact_dominant_ops[i].append(None)
                            elif o == 'IS_NOT_NULL':
                                fact_ops[i].append(col.ProjectOperator('>'))
                                fact_vals[i].append(col.ProjectValue(0))
                                fact_dominant_ops[i].append(
                                    col.ProjectOperatorDominant('>'))
                            else:
                                assert False, "Operator {} not supported".format(
                                    o)
                        else:
                            # No NULL values
                            if o == 'IS_NULL':
                                new_op = 'ALL_FALSE'
                            elif o == 'IS_NOT_NULL':
                                new_op = 'ALL_TRUE'
                            else:
                                assert False, "Operator {} not supported".format(
                                    o)
                            fact_ops[i].append(new_op)
                            fact_vals[i].append(None)  # This value is unused
                            fact_dominant_ops[i].append(None)
                    else:
                        # Handling =/<=/>=/</>.
                        # If the original column has a NaN, then we shoudn't
                        # include this in the result.  We can ensure this by
                        # adding a >0 predicate on the fact col.  Only need to
                        # do this if the original predicate is <, <=, or !=.
                        if o in ['<=', '<', '!='] and np.any(
                                pd.isnull(columns[j].all_distinct_values)):
                            fact_ops[i].append(col.ProjectOperator('>'))
                            fact_vals[i].append(col.ProjectValue(0))
                            fact_dominant_ops[i].append(
                                col.ProjectOperatorDominant('>'))
                        if v not in columns[j].all_distinct_values:
                            # Handle cases where value is not in the column
                            # vocabulary.
                            assert o in ['=', '!=']
                            if o == '=':
                                # Everything should be False.
                                fact_ops[i].append('ALL_FALSE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                            elif o == '!=':
                                # Everything should be True.
                                # Note that >0 has already been added,
                                # so there are no NULL results.
                                fact_ops[i].append('ALL_TRUE')
                                fact_vals[i].append(None)
                                fact_dominant_ops[i].append(None)
                        else:
                            # Distinct values of a factorized column is
                            # discretized.  So we do a lookup of the index for
                            # v in the original column, and not the fact col.
                            value = np.nonzero(
                                columns[j].all_distinct_values == v)[0][0]
                            p_v = col.ProjectValue(value)
                            p_op = col.ProjectOperator(o)
                            p_dom_op = col.ProjectOperatorDominant(o)
                            fact_ops[i].append(p_op)
                            fact_vals[i].append(p_v)
                            if p_dom_op in common.PROJECT_OPERATORS_DOMINANT.values(
                            ):
                                fact_dominant_ops[i].append(p_dom_op)
                            else:
                                fact_dominant_ops[i].append(None)

    assert len(fact_cols) == len(fact_ops) == len(fact_vals) == len(
        fact_dominant_ops)
    return fact_cols, fact_ops, fact_vals, fact_dominant_ops


def _infer_table_names(columns):
    ret = []
    for col in columns:
        if col.name.startswith('__in_') and col.factor_id in [None, 0]:
            ret.append(col.name[len('__in_'):])
    return ret


class ProgressiveSampling(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model,
            table,
            r,
            join_spec=None,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False,  # Skip sampling on wildcards?
            do_fanout_scaling=False,
    ):
        super(ProgressiveSampling, self).__init__()
        torch.set_grad_enabled(False)
        self.model = model
        self.model.eval()
        self.table = table
        self.all_tables = set(join_spec.join_tables
                             ) if join_spec is not None else _infer_table_names(
                                 table.columns)
        self.join_graph = join_spec.join_graph
        self.shortcircuit = shortcircuit
        self.do_fanout_scaling = do_fanout_scaling
        if do_fanout_scaling:
            self.num_tables = len(self.all_tables)
            print('# tables in join schema:', self.num_tables)

            # HACK: For convenience of running JOB-like bennchmarks; make
            # configurable.
            #
            # Additionally, we don't really need to know what is "primary", but
            # the code assumes so since it can be beneficial (e.g., don't learn
            # a bunch of trivial fanouts of 1).
            self.primary_table_name = 'title'

        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        ########### Inference optimizations below.
        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        # Important: enable this inference-time optimization only when model is
        # single-order.  When multiple orders, the masks would be updated so
        # don't cache a single version of "mask * weight".
        # TODO: turn this on only when benchmarking inference latency.
        # if 'MADE' in str(model) and self.model.num_masks == 1:
        #     for layer in model.net:
        #         if type(layer) == made.MaskedLinear:
        #             if layer.masked_weight is None:
        #                 layer.masked_weight = layer.mask * layer.weight
        #                 print('Setting masked_weight in MADE, do not retrain!')
        #         elif type(layer) == made.MaskedResidualBlock:
        #             for sublayer in layer.layers:
        #                 if type(sublayer) == made.MaskedLinear:
        #                     if sublayer.masked_weight is None:
        #                         sublayer.masked_weight = sublayer.mask * sublayer.weight
        #                         print(
        #                             'Setting masked_weight in MADE, do not retrain!'
        #                         )
        # for p in model.parameters():
        #     p.detach_()
        #     p.requires_grad = False
        # self.init_logits.detach_()

        self.logits_outs = []
        for natural_idx in range(self.model.nin):
            self.logits_outs.append(
                self.model.logits_for_col(natural_idx, self.init_logits))
        ########### END inference opts

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(self.num_samples, -1)

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def _maybe_remove_nan(self, dvs):
        # NOTE: "dvs[0] is np.nan" or "dvs[0] == np.nan" don't work.
        if dvs.dtype == np.dtype('object') and pd.isnull(dvs[0]):
            return dvs[1:], True
        return dvs, False

    def _get_valids(self, distinct_values, op, val, natural_idx, num_samples):
        """Returns a float vec indicating the valid bins in distinct_values."""
        distinct_values, removed_nan = self._maybe_remove_nan(distinct_values)
        valid_i = np.ones_like(distinct_values, np.bool)
        for o, v in zip(op, val):
            # Use &= since we assume conjunction.
            valid_i &= OPS[o](distinct_values, v)
        valid_i = valid_i.astype(np.float32, copy=False)
        if removed_nan:
            # NaN is always an invalid sample target, unless op is IS_NULL.
            v = 1. if op == ['IS_NULL'] else 0.
            valid_i = np.insert(valid_i, 0, v)
        return valid_i, False

    def _truncate_val_string(self, val):
        truncated_vals = []
        for v in val:
            if type(v) == tuple:
                new_val = str(list(v)[:20]) + '...' + str(
                    len(v) - 20) + ' more' if len(v) > 20 else list(v)
            else:
                new_val = v
            truncated_vals.append(new_val)
        return truncated_vals

    def _print_probs(self, columns, operators, vals, ordering, masked_logits):
        ml_i = 0
        for i in range(len(columns)):
            natural_idx = ordering[i]
            if operators[natural_idx] is None:
                continue
            truncated_vals = self._truncate_val_string(vals[natural_idx])
            print('  P({} {} {} | past) ~= {:.6f}'.format(
                columns[natural_idx].name, operators[natural_idx],
                truncated_vals, masked_logits[ml_i].mean().cpu().item()))
            ml_i += 1

    @profile
    def get_probs_for_col(self, logits, natural_idx, num_classes):
        """Returns probabilities for column i given model and logits."""
        num_samples = logits.size()[0]
        if False:  #self.model.UseDMoL(natural_idx):
            dmol_params = self.model.logits_for_col(
                natural_idx, logits)  # (num_samples, num_mixtures*3)
            logits_i = torch.zeros((num_samples, num_classes),
                                   device=self.device)
            for i in range(num_classes):
                logits_i[:, i] = distributions.dmol_query(
                    dmol_params,
                    torch.ones(num_samples, device=self.device) * i,
                    num_classes=num_classes,
                    num_mixtures=self.model.num_dmol,
                    scale_input=self.model.scale_input)
        else:
            logits_i = self.model.logits_for_col(
                natural_idx, logits, out=self.logits_outs[natural_idx])
        return torch.softmax(logits_i, 1)

    def _get_fanout_columns(self, cols, vals):
        # What tables are joined in this query?
        q_tables = GetTablesInQuery(cols, vals)
        some_query_table = next(iter(q_tables))  # pick any table in the query
        fanout_tables = self.all_tables - q_tables

        # For each table not in the query, find a path to q_table. The first
        # edge in the path gives the correct fanout column. We use
        # `shortest_path` here but the path is actually unique.
        def get_fanout_key(u):
            if self.join_graph is None:
                return None
            path = nx.shortest_path(self.join_graph, u, some_query_table)
            v = path[1]  # the first hop from the starting node u
            join_key = self.join_graph[u][v]["join_keys"][u]
            return join_key

        fanout_cols = [(t, get_fanout_key(t)) for t in fanout_tables]

        # The fanouts from a "primary" table are always 1.
        return list(
            filter(lambda tup: tup[0] != self.primary_table_name, fanout_cols))

    def _get_fanout_column_index(self, fanout_col):
        """Returns the natural index of a fanout column.

        For backward-compatibility, try both `__fanout_{table}` and
        `__fanout_{table}__{col}`.
        """
        table, key = fanout_col
        for col_name in [
                '__fanout_{}__{}'.format(table, key),
                '__fanout_{}'.format(table)
        ]:
            if col_name in self.table.name_to_index:
                return self.table.ColumnIndex(col_name)
        assert False, (fanout_col, self.table.name_to_index)

    @profile
    def _scale_probs(self, columns, operators, vals, p, ordering, num_fanouts,
                     num_indicators, inp):
        # TODO: Deal with indicators and fanout columns potentially being
        # factorized?  Current code assumes that these virtual columns are
        # never factorized (in JOB-M fanouts are factorized, but that's ok as
        # it *happens* to not need factorized fanouts in downscaling).

        # Find out what foreign tables are not present in this query.
        fanout_cols = self._get_fanout_columns(columns, vals)
        indexes_to_scale = [
            self._get_fanout_column_index(col) for col in fanout_cols
        ]

        if len(indexes_to_scale) == 0:
            return p.mean().item()

        # Make indexes_to_scale conform to sampling ordering.  No-op if
        # natural ordering is used.
        if isinstance(ordering, np.ndarray):
            ordering = list(ordering)
        sampling_order_for_scale = [
            ordering.index(natural_index) for natural_index in indexes_to_scale
        ]
        zipped = list(zip(sampling_order_for_scale, indexes_to_scale))
        zipped = sorted(zipped, key=lambda t: t[0])
        sorted_indexes_to_scale = list(map(lambda t: t[1], zipped))

        scale = 1.0
        for natural_index in sorted_indexes_to_scale:
            # Sample the fanout factors & feed them back as input.  This
            # modeling of AR dependencies among the fanouts improves errors
            # (than if they were just dependent on the content+indicators).
            logits = self._forward_encoded_input(inp,
                                                 sampling_ordering=ordering)

            # The fanouts are deterministic function based on join key.  We
            # can either model the join keys, and thus can sample a join
            # key and lookup.  Alternatively, we directly model the fanouts
            # and sample them.  Sample works better than argmax or
            # expectation.
            fanout_probs = self.get_probs_for_col(
                logits, natural_index, columns[natural_index].distribution_size)

            # Turn this on when measuring inference latency: multinomial() is
            # slightly faster than Categorical() then sample().  The latter has
            # a convenient method for printing perplexity though.
            # scales = torch.multinomial(fanout_probs, 1)
            dist = torch.distributions.categorical.Categorical(fanout_probs)
            scales = dist.sample()

            # Off-by-1 in fanout's domain: 0 -> np.nan, 1 -> value 0, 2 ->
            # value 1.
            actual_scale_values = (scales - 1).clamp_(1)

            if 'VERBOSE' in os.environ:
                print('scaling', columns[natural_index], 'with',
                      actual_scale_values.float().mean().item(), '; perplex.',
                      dist.perplexity()[:3])

            scale *= actual_scale_values

            # Put the sampled 'scales' back into input, so that fanouts to
            # be sampled next can depend on the current fanout value.
            self._put_samples_as_input(scales.view(-1, 1), inp, natural_index)

        if os.environ.get('VERBOSE', None) == 2:
            print('  p quantiles',
                  np.quantile(p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]), 'mean',
                  p.mean())
            print('  scale quantiles',
                  np.quantile(scale.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]),
                  'mean', scale.mean())

        scaled_p = p / scale.to(torch.float)

        if os.environ.get('VERBOSE', None) == 2:
            print('  scaled_p quantiles',
                  np.quantile(scaled_p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]),
                  'mean', scaled_p.mean())

        # NOTE: overflow can happen for undertrained models.
        scaled_p[scaled_p == np.inf] = 0

        if os.environ.get('VERBOSE', None) == 2:
            print('  (after clip) scaled_p quantiles',
                  np.quantile(scaled_p.cpu().numpy(), [0.5, 0.9, 0.99, 1.0]),
                  'mean', scaled_p.mean())
        return scaled_p.mean().item()

    @profile
    def _put_samples_as_input(self,
                              data_to_encode,
                              inp,
                              natural_idx,
                              sampling_order_idx=None):
        """Puts [bs, 1] sampled values approipately into inp."""
        if not isinstance(self.model, transformer.Transformer):
            if natural_idx == 0:
                self.model.EncodeInput(
                    data_to_encode,
                    natural_col=0,
                    out=inp[:, :self.model.input_bins_encoded_cumsum[0]])
            else:
                l = self.model.input_bins_encoded_cumsum[natural_idx - 1]
                r = self.model.input_bins_encoded_cumsum[natural_idx]
                self.model.EncodeInput(data_to_encode,
                                       natural_col=natural_idx,
                                       out=inp[:, l:r])
        else:
            # Transformer.  Need special treatment due to
            # right-shift.
            l = (natural_idx + 1) * self.model.d_model
            r = l + self.model.d_model
            if sampling_order_idx == 0:
                # Let's also add E_pos=0 to SOS (if enabled).
                # This is a no-op if disabled pos embs.
                self.model.EncodeInput(
                    data_to_encode,  # Will ignore.
                    natural_col=-1,  # Signals SOS.
                    out=inp[:, :self.model.d_model])

            if transformer.MASK_SCHEME == 1:
                # Should encode natural_col \in [0, ncols).
                self.model.EncodeInput(data_to_encode,
                                       natural_col=natural_idx,
                                       out=inp[:, l:r])
            elif natural_idx < self.model.nin - 1:
                # If scheme is 0, should not encode the last
                # variable.
                self.model.EncodeInput(data_to_encode,
                                       natural_col=natural_idx,
                                       out=inp[:, l:r])

    @profile
    def _forward_encoded_input(self, inp, sampling_ordering):
        if hasattr(self.model, 'do_forward'):
            if isinstance(self.model, made.MADE):
                inv = utils.InvertOrder(sampling_ordering)
                logits = self.model.do_forward(inp, inv)
            else:
                logits = self.model.do_forward(inp, sampling_ordering)
        else:
            if self.traced_fwd is not None:
                logits = self.traced_fwd(inp)
            else:
                logits = self.model.forward_with_encoded_input(inp)
        return logits

    @profile
    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  inp=None):
        torch.set_grad_enabled(False)
        ncols = len(columns)
        logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []
        valid_i_list = [None] * ncols

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]
            if i != 0:
                num_i = 1
            else:
                num_i = num_samples if num_samples else int(
                    self.r * self.dom_sizes[natural_idx])
            if self.shortcircuit and operators[natural_idx] is None:
                self._put_samples_as_input(None, inp, natural_idx)
                data_to_encode = None
            else:
                if operators[natural_idx] is not None:
                    dvs = columns[natural_idx].all_distinct_values
                    valid_i, has_mask = self._get_valids(
                        dvs, operators[natural_idx], vals[natural_idx],
                        natural_idx, num_samples)

                # This line triggers a host -> gpu copy, showing up as a
                # hotspot in cprofile.
                valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)
                num_classes = len(dvs)
                probs_i = self.get_probs_for_col(logits,
                                                 natural_idx,
                                                 num_classes=num_classes)

                valid_i = valid_i_list[i]
                if valid_i is not None:
                    probs_i *= valid_i
                probs_i_summed = probs_i.sum(1)
                masked_probs.append(probs_i_summed)

                if i == ncols - 1:
                    break

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)

                samples_i = torch.multinomial(probs_i,
                                              num_samples=num_i,
                                              replacement=True)  # [bs, num_i]

                # Make sure to revert the probabilities so that the final
                # calculation is correct.
                probs_i = probs_i.masked_fill_(paths_vanished, 0.0)
                # Only Factorized PS should go down this path.
                if has_mask:
                    self.update_factor_mask(samples_i.view(-1,),
                                            vals[natural_idx], natural_idx)

                data_to_encode = samples_i.view(-1, 1)

                # Encode input: i.e., put sampled vars into input buffer.
                if data_to_encode is not None:  # Wildcards are encoded already.
                    self._put_samples_as_input(data_to_encode,
                                               inp,
                                               natural_idx,
                                               sampling_order_idx=i)

            if i == ncols - 1:
                break

            # Actual forward pass.
            next_natural_idx = i + 1 if ordering is None else ordering[i + 1]
            if self.shortcircuit and operators[next_natural_idx] is None:
                # If next variable in line is wildcard, then don't do
                # this forward pass.  Var 'logits' won't be accessed.
                continue
            logits = self._forward_encoded_input(inp,
                                                 sampling_ordering=ordering)
        # Debug outputs.
        if 'VERBOSE' in os.environ:
            self._print_probs(columns, operators, vals, ordering, masked_probs)

        if len(masked_probs) > 1:
            # Doing this convoluted scheme because m_l[0] is a scalar, and
            # we want the correct shape to broadcast.
            p = masked_probs[1]
            for ls in masked_probs[2:]:
                p *= ls
            p *= masked_probs[0]
        else:
            p = masked_probs[0]

        if self.do_fanout_scaling:
            num_fanouts = self.num_tables - 1
            num_indicators = self.num_tables

            return self._scale_probs(columns, operators, vals, p, ordering,
                                     num_fanouts, num_indicators, inp)

        return p.mean().item()

    def _StandardizeQuery(self, columns, operators, vals):
        return FillInUnqueriedColumns(self.table, columns, operators, vals)

    def Query(self, columns, operators, vals):
        torch.cuda.empty_cache()

        # Massages queries into natural order.
        columns, operators, vals = self._StandardizeQuery(
            columns, operators, vals)

        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        with torch.no_grad():
            inp_buf = self.inp.zero_()

            if num_orderings == 1:
                ordering = orderings[0]
                inv_ordering = utils.InvertOrder(ordering)
                self.OnStart()
                p = self._sample_n(
                    self.num_samples,
                    # MADE's 'orderings', 'm[-1]' are in an inverted space
                    # --- opposite semantics of what _sample_n() and what
                    # Transformer's ordering expect.
                    ordering if isinstance(self.model, transformer.Transformer)
                    else inv_ordering,
                    columns,
                    operators,
                    vals,
                    inp=inp_buf)
                self.OnEnd()

                if os.environ.get('VERBOSE', None) == 2:
                    print('density={}'.format(p))
                    print('scaling with', self.cardinality, '==',
                          p * self.cardinality)

                return np.ceil(p * self.cardinality).astype(dtype=np.int64,
                                                            copy=False)

            # Num orderings > 1.
            ps = []
            self.OnStart()
            for ordering in orderings:
                ordering = ordering if isinstance(
                    self.model,
                    transformer.Transformer) else utils.InvertOrder(ordering)
                ns = self.num_samples // num_orderings
                if ns < 1:
                    print("WARNING: rounding up to 1", self.num_samples,
                          num_orderings)
                    ns = 1
                p_scalar = self._sample_n(ns, ordering, columns, operators,
                                          vals)
                ps.append(p_scalar)
            self.OnEnd()

            if np.mean(ps) == np.inf:
                # Sometimes happens for under-trained models.
                print('WARNING: clipping underflows to 0;', ps)
                ps = np.nan_to_num(ps, posinf=0)

            print('np.mean(ps)', np.mean(ps), 'self.cardinality',
                  self.cardinality, 'prod',
                  np.mean(ps) * self.cardinality, 'ceil',
                  np.ceil(np.mean(ps) * self.cardinality))
            return np.ceil(np.mean(ps) * self.cardinality).astype(
                dtype=np.int64, copy=False)


class FactorizedProgressiveSampling(ProgressiveSampling):
    """Additional logic for handling column factorization."""

    def __init__(self,
                 model,
                 fact_table,
                 r,
                 join_spec=None,
                 device=None,
                 seed=False,
                 cardinality=None,
                 shortcircuit=False,
                 do_fanout_scaling=None):
        self.fact_table = fact_table
        self.base_table = fact_table.base_table
        self.factor_mask = None

        super(FactorizedProgressiveSampling,
              self).__init__(model, fact_table, r, join_spec, device, seed,
                             cardinality, shortcircuit, do_fanout_scaling)

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.base_table.columns[0].DistributionSize())
        return 'fact_psample_{}'.format(n)

    def _StandardizeQuery(self, columns, operators, vals):
        self.original_query = (columns, operators, vals)
        cols, ops, vals = FillInUnqueriedColumns(self.base_table, columns,
                                                 operators, vals)
        cols, ops, vals, dominant_ops = ProjectQuery(self.fact_table, cols, ops,
                                                     vals)
        self.dominant_ops = dominant_ops
        return cols, ops, vals

    @profile
    def _get_valids(self, distinct_values, op, val, natural_idx, num_samples):
        """Returns a valid mask of shape (), (size(col)) or (N, size(col)).

        For columns that are not factorized, the first dimension is trivial.
        For columns that are not filtered, both dimensions are trivial.
        """
        # Indicates whether valid values for this column depends on samples
        # from previous columns.  Only used for factorized columns with
        # >/</>=/<=/!=/IN.By default this is False.
        has_mask = False
        # Column i.
        if op is not None:
            # There exists a filter.
            if self.fact_table.columns[natural_idx].factor_id is None:
                # This column is not factorized.
                distinct_values, removed_nan = self._maybe_remove_nan(
                    distinct_values)
                valids = [OPS[o](distinct_values, v) for o, v in zip(op, val)]
                valid = np.logical_and.reduce(valids, 0).astype(np.float32,
                                                                copy=False)
                if removed_nan:
                    # NaN is always an invalid sample target, unless op is
                    # IS_NULL.
                    v = 1. if op == ['IS_NULL'] else 0.
                    valid = np.insert(valid, 0, v)
            else:
                # This column is factorized.  `valid` stores the valid values
                # for this column for each operator.  At the very end, combine
                # the valid values for the operatorsvia logical and.
                valid = np.ones((len(op), len(distinct_values)), np.bool)
                for i, (o, v) in enumerate(zip(op, val)):
                    # Handle the various operators.  For ops with a mask, we
                    # add a new dimension so that we can add the mask.  Refer
                    # to `update_factor_mask` for description
                    # ofself.factor_mask.
                    if o in common.PROJECT_OPERATORS.values(
                    ) or o in common.PROJECT_OPERATORS_LAST.values():
                        valid[i] &= OPS[o](distinct_values, v)
                        has_mask = True
                        if self.fact_table.columns[natural_idx].factor_id > 0:
                            if len(valid.shape) != 3:
                                valid = np.tile(np.expand_dims(valid, 1),
                                                (1, num_samples, 1))
                            assert valid.shape == (len(op), num_samples,
                                                   len(distinct_values))
                            expanded_mask = np.expand_dims(
                                self.factor_mask[i], 1)
                            assert expanded_mask.shape == (num_samples, 1)
                            valid[i] |= expanded_mask
                    # IN is special case.
                    elif o == 'IN':
                        has_mask = True
                        v_list = np.array(list(v))
                        matches = distinct_values[:, None] == v_list
                        assert matches.shape == (len(distinct_values),
                                                 len(v_list)), matches.shape
                        if self.fact_table.columns[natural_idx].factor_id > 0:
                            if len(valid.shape) != 3:
                                valid = np.tile(np.expand_dims(valid, 1),
                                                (1, num_samples, 1))
                            assert valid.shape == (
                                len(op), num_samples,
                                len(distinct_values)), valid.shape
                            matches = np.tile(matches, (num_samples, 1, 1))
                            expanded_mask = np.expand_dims(
                                self.factor_mask[i], 1)
                            matches &= expanded_mask
                        valid[i] = np.logical_or.reduce(
                            matches, axis=-1).astype(np.float32, copy=False)
                    else:
                        valid[i] &= OPS[o](distinct_values, v)
                valid = np.logical_and.reduce(valid, 0).astype(np.float32,
                                                               copy=False)
                assert valid.shape == (num_samples,
                                       len(distinct_values)) or valid.shape == (
                                           len(distinct_values),), valid.shape
        else:
            # This column is unqueried.  All values are valid.
            valid = 1.0

        # Reset the factor mask if this col is not factorized
        # or if this col is the first subvar
        # or if we don't need to maintain a mask for this predicate.
        if self.fact_table.columns[natural_idx].factor_id in [None, 0
                                                             ] or not has_mask:
            self.factor_mask = None

        return valid, has_mask

    @profile
    def update_factor_mask(self, s, val, natural_idx):
        """Updates the factor mask for the next iteration.

        Factor mask is a list of length len(ops).  Each element in the list is
        a numpy array of shape (N, ?)  where the second dimension can be
        different sizes for different operators, indicating where a previous
        factor dominates the remaining for a column.

        We keep a separate mask for each operator for cases where there are
        multiple conditions on the same col. In these cases, there can be
        situations where a subvar dominates or un-dominates for one condition
        but not for others.

        The reason we need special handling is for cases like this: Let's say
        we have factored column x = (x1, x2) and literal y = (y1, y2) and the
        predicate is x > y.  By default we assume conjunction between subvars
        and so x>y would be treated as x1>y1 and x2>y2, which is incorrect.

        The correct handling would be
        x > y iff
            (x1 >= y1 and x2 > y2) OR
            x1 > y1.
        Because x1 contains the most significant bits, if x1>y1, then x2 can be
        anything.

        For the general case where x is factorized as (x1...xN) and y is
        factorized as (y1...yN),
        x > y iff
            (x1 >= y1 and x2 >= y2 and ... xN > yN) OR
            x1 > y1 OR x2 > y2 ... OR x(N-1) > y(N-1).
        To handle this, as we perform progressive sampling, we apply a
        "dominant operator" (> for this example) to the samples, and keep a
        running factor_mask that gets OR'd for calculating future valid
        vals. This function handles this.

        Other Examples:
            factored column x = (x1, x2), literals y = (y1, y2), z = (z1, z2)
            x < y iff
                (x1 <= y1 and x2 < y2) OR
                x1 < y1.
            x >= y iff
                (x1 >= y1 and x2 >= y2) OR
                x1 > y1.
            x <= y iff
                (x1 <= y1 and x2 <= y2) OR
                x1 < y1.
            x != y iff
                (any(x1) and x2 != y2) OR
                x1 != y1

        IN predicates are handled differently because instead of a single
        literal, there is a list of values. For example,
        x IN [y, z] if
                (x1 == y1 and x2 == y2) OR
                (x1 == z1 and x2 == z2)

        This function is called after _get_valids().  Note that _get_valids()
        would access self.factor_mask only after the first subvar for a given
        var (natural_idx).  By that time self.factor_mask has been assigned
        during the first invocation of this function.  Field self.factor_mask
        is reset to None by _get_valids() whenever we move to a new
        original-space column, or when the column has no operators that need
        special handling.
        """
        s = s.cpu().numpy()
        if self.factor_mask is None:
            self.factor_mask = [None] * len(self.dominant_ops[natural_idx])
        for i, (p_op_dominant,
                v) in enumerate(zip(self.dominant_ops[natural_idx], val)):
            if p_op_dominant == 'IN':
                # Mask for IN should be size (N, len(v))
                v_list = list(v)
                new_mask = s[:, None] == v_list
                # In the example above, for each sample s_i,
                # new_mask stores
                # [
                #   (s_i_1 == y1 and s_i_2 == y2 ...),
                #   (s_i_1 == z1 and s_i_2 == z2 ...),
                #   ...
                # ]
                # As we sample, we &= the current mask with previous masks.
                assert new_mask.shape == (len(s), len(v_list)), new_mask.shape
                if self.factor_mask[i] is not None:
                    new_mask &= self.factor_mask[i]
            elif p_op_dominant in common.PROJECT_OPERATORS_DOMINANT.values():
                new_mask = OPS[p_op_dominant](s, v)
                if self.factor_mask[i] is not None:
                    new_mask |= self.factor_mask[i]
            else:
                assert p_op_dominant is None, 'This dominant operator ({}) is not supported.'.format(
                    p_op_dominant)
                new_mask = np.zeros_like(s, dtype=np.bool)
                if self.factor_mask[i] is not None:
                    new_mask |= self.factor_mask[i]
            self.factor_mask[i] = new_mask


class JoinSampling(CardEst):
    """Draws tuples using a join sampler.

    Different from ProgressiveSampling, such methods are unconditional: they
    cannot draw in-query region tuples that satisfy some constraints.
    """

    def __init__(self, join_iter, table, num_samples):
        super(JoinSampling, self).__init__()
        assert isinstance(join_iter, common.SamplerBasedIterDataset), join_iter

        # Make sure all columns are disambiguated.
        def _disambiguate(table):
            for col in table.columns:
                if ':' not in col.name:
                    col.name = common.JoinTableAndColumnNames(table.name,
                                                              col.name,
                                                              sep=':')
            # For the pd.DataFrame header as well.
            table.data.columns = [col.name for col in table.columns]
            return table

        self.table_dict = dict(
            (t.name, _disambiguate(t)) for t in join_iter.tables)

        join_keys_dict = datasets.JoinOrderBenchmark.GetJobLightJoinKeys()
        self.join_keys_dict = dict(
            (t_name, common.JoinTableAndColumnNames(t_name, c_name, sep=':'))
            for t_name, c_name in join_keys_dict.items())

        self.join_iter = join_iter
        self.table = table
        self.num_samples = num_samples
        self.sample_durs_ms = []
        self.rng = np.random.RandomState(1234)

    def __str__(self):
        return '{}_{}'.format(type(self.join_iter).__name__, self.num_samples)

    def _StandardizeQuery(self, columns, operators, vals):
        return FillInUnqueriedColumns(self.table, columns, operators, vals)

    def _accumulate_samples(self, columns, operators, vals):
        """Samples from the inner join of each query graph."""

        table_names = list(GetTablesInQuery(columns, vals))
        table_names.remove('title')
        table_names = ['title'] + table_names

        sampler = 'factorized_sampler'  # For JOB-M.
        tables_key = query_lib.MakeTablesKey(table_names)

        tables_in_templates = [self.table_dict[n] for n in table_names]

        def _FilterJoinConds(table_names, join_clauses):
            relevant_clauses = []
            for line in join_clauses:
                groups = join_utils.match_join_clause_or_fail(line).groups()
                t1, t2 = groups[0], groups[2]
                if t1 in table_names and t2 in table_names:
                    relevant_clauses.append(line)
            return relevant_clauses

        def _DifferentJoinGraph(self, join_clauses):
            if not hasattr(self, 'last_join_clauses'):
                return True
            jcs = sorted(join_clauses)
            return not np.array_equal(self.last_join_clauses, jcs)

        def _CacheJoinGraph(self, join_clauses):
            self.last_join_clauses = sorted(join_clauses)

        import experiments
        experiment = experiments.JOB_M  # TODO: make configurable.
        join_root = 'title'
        join_clauses = _FilterJoinConds(table_names, experiment['join_clauses'])
        config = {
            # For retaining a specially crafted order (bottom-up).
            'join_tables': list(
                filter(lambda t: t in table_names, experiment['join_tables'])),
            'join_keys': experiment['join_keys'],
            'join_clauses': join_clauses,
            'join_root': join_root,
            'join_how': 'inner',
        }
        join_spec = join_utils.get_join_spec(config)
        print('Join spec', join_spec)

        if _DifferentJoinGraph(self, join_clauses):
            # Needed because current FactorizedSampler impl doesn't write
            # out disambiguation info in file names (./cache/*).
            print('Removing ./cache/')
            shutil.rmtree('./cache/')

        s = factorized_sampler.FactorizedSampler(
            tables_in_templates, join_spec, sample_batch_size=self.num_samples)

        _CacheJoinGraph(self, join_clauses)

        sample_df = s.run()

        # 'title:title:id', 'title:title:kind_id'
        print('before renaming:', sample_df.columns)
        rename = {
            c: ':'.join(c.split(':')[-2:]) if ':' in c else c
            for c in sample_df.columns
        }
        sample_df.rename(rename, axis=1, inplace=True)
        print('after renaming:', sample_df.columns)
        inner_join_cardinality = s.jct_actors[join_root].jct['{}.weight'.format(
            join_root)].sum()
        return sample_df, inner_join_cardinality

    def _null_safe_get_valids(self, data_df, op, val):
        if op == 'IS_NULL':
            return pd.isnull(data_df)
        overall_valid = np.zeros(len(data_df), dtype=np.bool)
        isnull = pd.isnull(data_df)
        non_null_portion = data_df[~isnull]
        non_null_valid = OPS[op](non_null_portion, val)
        overall_valid[~isnull] = non_null_valid
        return overall_valid

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        columns, operators, vals = self._StandardizeQuery(
            columns, operators, vals)
        self.OnStart()
        sampled_df, norm_const = self._accumulate_samples(
            columns, operators, vals)
        valid_masks = []
        for col, op, val in zip(columns, operators, vals):
            if op is None:
                continue
            for o, v in zip(op, val):
                # If we sample from the query graph's inner join, by
                # definition they are already in query graph.  Skip
                # indicator constraints.
                if not col.name.startswith('__in_'):
                    print('sampled_df.columns', sampled_df.columns)
                    valid_masks.append(
                        self._null_safe_get_valids(sampled_df[col.name], o, v))
        s = np.all(valid_masks, axis=0).sum()
        sel = s * 1.0 / self.num_samples
        print('sel', sel, 'valid samples', s, 'normalizing constant',
              norm_const)
        self.OnEnd()
        return np.ceil(sel * norm_const).astype(dtype=np.int64)


class Postgres(CardEst):

    def __init__(self, database, table_names, port=None):
        """Postgres estimator (i.e., EXPLAIN).  Must have the PG server live.
        E.g.,
            def MakeEstimators():
                return [Postgres('dmv', 'vehicle_reg', None), ...]
        Args:
          database: string, the database name.
          table_names: List[string], list of table names
          port: int, the port.
        """
        import psycopg2

        super(Postgres, self).__init__()

        self.conn = psycopg2.connect(database=database, port=port)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self.name = 'Postgres'

        for relation in table_names:
            self.cursor.execute('analyze ' + relation + ';')
            self.conn.commit()
        self.table_names = table_names

        self.database = database

    def __str__(self):
        return 'postgres'

    def QuerySql(self, sql_query):
        """
        Args:
            sql_query (string): sql query to run
        """
        sql_query = sql_query.replace('COUNT(*)', '*')
        sql_query = sql_query.replace('SELECT(*)', 'SELECT *')
        query_s = 'explain(format json) ' + sql_query
        # print(query_s)
        self.OnStart()
        self.cursor.execute(query_s)
        res = self.cursor.fetchall()
        # print(res)
        result = res[0][0][0]['Plan']['Plan Rows']
        self.OnEnd()
        return result

    def QueryByExecSql(self, sql_query):
        query_s = sql_query
        self.string = query_s

        self.cursor.execute(query_s)
        result = self.cursor.fetchone()[0]

        return result

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        pred = QueryToPredicate(columns, operators, vals)
        # Use json so it's easier to parse.
        query_s = 'select * from ' + self.relation + pred
        return self.QuerySql(query_s)

    def QueryByExec(self, columns, operators, vals):
        # Runs actual query on postgres and returns true cardinality.
        assert len(columns) == len(operators) == len(vals)

        pred = QueryToPredicate(columns, operators, vals)
        query_s = 'select count(*) from ' + self.relation + pred
        return self.QueryByExecSql(query_s)

    def Close(self):
        self.cursor.close()
        self.conn.close()
