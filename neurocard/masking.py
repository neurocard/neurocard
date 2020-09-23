"""Masking module: random input masking, table masking, etc."""
import numpy as np

import torch
import torch.nn as nn

import common


# TODO: refactor made.py to use this class.
class Masking(object):
    """Column masking logic."""

    @classmethod
    def Params(cls):
        p = {}
        p['draw_dropout_per_col'] = False
        p['per_row_dropout'] = False
        return p

    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)

        # Optimization.
        self._constant_ones_cache = {}

    def input_mask(self, x, is_training):
        """Calculates a random input mask for training.

        Args:
          x: input tokens, shaped [batch_size, num_cols].
          is_training: bool.

        Returns:
          batch_mask, bools, shaped [batch_size, num_cols, 1], where 1 means
            use original representation and 0 means use MASK representaions.
            During inference this should be all 1s.
        """
        assert x.ndim == 2, x.shape

        if not is_training:
            # During inference, short-circuit to immediately return 1s.
            return torch.ones_like(x).unsqueeze(2)

        if self.table_dropout:
            return self._table_dropout(x, is_training)

        return self._vanilla_dropout(x, is_training)

    def _get_cached_constant_ones(self, shape, device):
        """Returns a cached all-one tensor with desired shape and device."""
        key = (shape, device)
        if key not in self._constant_ones_cache:
            self._constant_ones_cache[key] = torch.ones(*shape, device=device)
        return self._constant_ones_cache[key]

    def _table_dropout(self, x, is_training):
        bs, inp_seq_len = x.shape

        kOnes = self._get_cached_constant_ones((bs, 1), x.device)

        if self.per_row_dropout:
            # NOTE: torch.rand* funcs on GPU are ~4% slower than
            # generating them on CPU via np.random.
            num_dropped_tables = np.random.randint(1, self.num_joined_tables,
                                                   (bs, 1)).astype(np.float32,
                                                                   copy=False)
            table_dropped = np.random.rand(bs, self.num_joined_tables) <= (
                num_dropped_tables / self.num_joined_tables)
            if self.table_primary_index is not None:
                table_dropped[:, self.table_primary_index] = False
            normal_drop_rands = np.random.rand(bs, inp_seq_len)
            table_dropped = table_dropped.astype(np.float32, copy=False)
        else:
            # 1 means drop that table.
            num_dropped_tables = np.random.randint(1, self.num_joined_tables)
            table_dropped = np.random.rand(
                self.num_joined_tables
            ) <= num_dropped_tables / self.num_joined_tables
            if self.table_primary_index is not None:
                table_dropped[self.table_primary_index] = False
            table_dropped = table_dropped.astype(np.float32, copy=False)

        batch_masks = []
        for i in range(inp_seq_len):
            # Table dropout.  Logic:
            #  First, draw the tables to be dropped.
            #  If a table T is dropped:
            #    Drop its content columns & indicator only.
            #    Don't drop its fanout.
            #  Otherwise:
            #    Uniformly wraw # content columns to drop.
            #    Don't drop its indicator.
            #    Drop its fanout.
            table_index = self.table_indexes[i]

            if self.per_row_dropout:
                # table_dropped[table_index]: shaped [BS, 1]
                #   elem 0 : True
                #   elem 1 : True
                #   elem 2 : False, etc.
                is_content = float(
                    self.table_column_types[i] == common.TYPE_NORMAL_ATTR)
                is_fanout = float(
                    self.table_column_types[i] == common.TYPE_FANOUT)
                use_unk = table_dropped[:, table_index]
                if is_fanout:
                    # Column i is a fanout column.  Drop iff table not dropped.
                    batch_mask = torch.tensor(use_unk).float().unsqueeze(1).to(
                        x.device)
                else:
                    # Handle batch elements where this table is not dropped.
                    normal_drop_prob = np.random.randint(
                        0, self.table_num_columns[table_index] + 1,
                        (bs,)) * 1. / self.table_num_columns[table_index]

                    normal_drop = normal_drop_rands[:, i] <= normal_drop_prob
                    # Make sure we drop content only.
                    normal_drop = normal_drop * is_content

                    not_dropped_pos = (use_unk == 0.0)
                    use_unk[not_dropped_pos] = normal_drop[not_dropped_pos]

                    # Shaped [bs, 1].
                    batch_mask = torch.as_tensor(1.0 - use_unk).unsqueeze(1).to(
                        x.device)

            else:
                # Make decisions for entire batch.
                if table_dropped[table_index]:
                    # Drop all its normal attributes + indicator.  Don't drop
                    # fanout.
                    batch_mask = torch.clamp(
                        torch.dropout(
                            kOnes,
                            p=1.0 -
                            (self.table_column_types[i] == common.TYPE_FANOUT),
                            train=is_training), 0, 1)

                else:
                    # Drop each normal attribute with drawn propability.
                    # Don't drop indicator.
                    # Drop fanout.
                    drop_p = 0.0
                    if self.table_column_types[i] == common.TYPE_NORMAL_ATTR:
                        # Possible to drop all columns of this
                        # table (it participates in join but no
                        # attributes are filtered).
                        drop_p = np.random.randint(
                            0, self.table_num_columns[table_index] +
                            1) / self.table_num_columns[table_index]
                    elif self.table_column_types[i] == common.TYPE_FANOUT:
                        drop_p = 1.0
                    batch_mask = torch.clamp(
                        torch.dropout(kOnes, p=drop_p, train=is_training), 0, 1)
            batch_masks.append(batch_mask)

        # [bs, num cols, 1].
        return torch.cat(batch_masks, 1).unsqueeze(-1)

    def _vanilla_dropout(self, x, is_training):
        bs, inp_seq_len = x.shape

        if self.draw_dropout_per_col:
            kOnes = self._get_cached_constant_ones((bs, 1, 1), x.device)
            vecs = []
            for _ in range(inp_seq_len):
                vecs.append(
                    torch.dropout(kOnes,
                                  p=np.random.randint(0, inp_seq_len) /
                                  inp_seq_len,
                                  train=is_training))
            dropout_vec = torch.cat(vecs, dim=1)
        else:
            kOnes = self._get_cached_constant_ones((bs, inp_seq_len, 1),
                                                   x.device)
            dropout_vec = torch.dropout(kOnes,
                                        p=np.random.randint(0, inp_seq_len) /
                                        inp_seq_len,
                                        train=is_training)
        # During training, non-dropped 1's are scaled by 1/(1-p), so we
        # clamp back to 1.  Shaped [bs, num cols, 1].
        batch_mask = torch.clamp(dropout_vec, 0, 1)
        return batch_mask
