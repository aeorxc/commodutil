import os
import unittest

import pandas as pd

from commodutil import forwards
from commodutil import stats


class TestForwards(unittest.TestCase):
    def test_curve_zscore(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )
        contracts = cl.rename(
            columns={
                x: pd.to_datetime(forwards.convert_contract_to_date(x))
                for x in cl.columns
            }
        )
        hist = contracts[["2020-01-01"]].dropna()

        fwd = contracts[["2020-01-01"]]

        res = stats.curve_seasonal_zscore(hist, fwd)

        self.assertAlmostEqual(res["zscore"]["2019-01-02"], 0.92, 2)

    def test_reindex_zscore(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )
        contracts = cl.rename(
            columns={
                x: pd.to_datetime(forwards.convert_contract_to_date(x))
                for x in cl.columns
            }
        )

        q = forwards.quarterly_contracts(contracts)
        q = q[[x for x in q.columns if "Q1" in x]]

        res = stats.reindex_zscore(q, calc_year_start=2022)
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
