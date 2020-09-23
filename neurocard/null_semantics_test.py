"""Tests numpy null semantics.

For any value (e.g., float, int, or np.nan) v:

  np.nan <op> v == False.

This means in progressive sampling, if a column's domain contains np.nan (at
the first position in the domain), it will never be a valid sample target.

Consistent with Postgres semantics:
  select min(production_year) from title as t;  -- 1880
  select count(*) from title as t where t.production_year < 1880;  -- 0
  select count(*) from title as t where t.production_year is null; -- 72094
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import pandas as pd

import datasets


class NullSemanticsTest(absltest.TestCase):

    def testSimple(self):
        t = datasets.LoadImdb('title')
        self.assertEqual('production_year', t.columns[-1].name)
        production_year = t.columns[-1]

        self.assertTrue(pd.isnull(production_year.all_distinct_values[0]))
        min_val = production_year.all_distinct_values[1]
        # 'RuntimeWarning: invalid value encountered in less' expected.
        s = (production_year.all_distinct_values < min_val).sum()
        self.assertEqual(0, s, 'np.nan should not be considered as < value')

        s = (production_year.all_distinct_values == np.nan).sum()
        self.assertEqual(0, s, 'np.nan not == np.nan either')


if __name__ == '__main__':
    absltest.main()
