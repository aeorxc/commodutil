import os
import unittest
import pytest
import numpy as np
import pandas as pd

from commodutil import pandasutil
from commodutil.forward.util import convert_contract_to_date


def _fillna_downbet_oracle(df):
    """Original loop-based implementation, kept inline as the parity oracle."""
    df = df.copy()
    for col in df.columns:
        non_nans = df[col].dropna()
        if len(non_nans) > 1:
            start, end = non_nans.index[0], non_nans.index[-1]
            df.loc[start:end, col] = df.loc[start:end, col].ffill()
    return df


class TestPandasUtils(unittest.TestCase):
    def test_mergets(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            date_format="%Y-%m-%d",
        )
        contracts = cl.rename(
            columns={x: pd.to_datetime(convert_contract_to_date(x)) for x in cl.columns}
        )

        res = pandasutil.mergets(
            contracts["2020-01-01"],
            contracts["2020-02-01"],
            leftl="Test1",
            rightl="Test2",
        )
        self.assertIn("Test1", res.columns)
        self.assertIn("Test2", res.columns)

    def test_sql_insert(self):
        df = pd.DataFrame(
            [[1, 2, 3], [4, "test'ing", 6], [7, 8, 9]], columns=["a", "b", "c"]
        )
        res = pandasutil.sql_insert_statement_from_dataframe(df, "table")
        exp = "INSERT INTO table (a, b, c) VALUES (1, 2, 3)"
        self.assertEqual(res[0], exp)
        exp = "INSERT INTO table (a, b, c) VALUES (4, 'testing', 6)"
        self.assertEqual(res[1], exp)

    def test_fillna_downbet_parity(self):
        # Frame exercising leading, interior and trailing NaN gaps, plus an
        # all-NaN column and a single-non-NaN-value column.
        df = pd.DataFrame(
            {
                # leading NaNs, interior gap, trailing NaNs
                "a": [np.nan, np.nan, 1.0, np.nan, np.nan, 4.0, np.nan],
                # interior gap only
                "b": [10.0, np.nan, np.nan, 13.0, 14.0, np.nan, 16.0],
                # all NaN
                "c": [np.nan] * 7,
                # single non-NaN value (hits the old `len(non_nans) > 1` guard)
                "d": [np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan],
                # single non-NaN value at the end
                "e": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9.0],
                # fully populated
                "f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )

        expected = _fillna_downbet_oracle(df)
        result = pandasutil.fillna_downbet(df)

        pd.testing.assert_frame_equal(result, expected)
        # input must not be mutated
        self.assertEqual(df["a"].isna().sum(), 5)

    def test_apply_formula(self):
        data = {
            "BRN": [85.14, 86.24, 85.34, 86.17, 87.55],
            "G": [899.50, 903.50, 889.00, 888.75, 941.75],
        }
        df = pd.DataFrame(data)
        res = pandasutil.apply_formula(df, "G/7.45-BRN")
        self.assertEqual(res.iloc[0, 0], pytest.approx(35.598, abs=0.01))
        self.assertEqual(res.iloc[-1, 0], pytest.approx(38.859, abs=0.01))


if __name__ == "__main__":
    unittest.main()
