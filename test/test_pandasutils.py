import os
import unittest
import pytest
import pandas as pd

from commodutil import forwards
from commodutil import pandasutil


class TestPandasUtils(unittest.TestCase):
    def test_mergets(self):
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

    def test_apply_formula(self):
        data = {
            "BRN": [85.14, 86.24, 85.34, 86.17, 87.55],
            "G": [899.50, 903.50, 889.00, 888.75, 941.75],
        }
        df = pd.DataFrame(data)
        res = pandasutil.apply_formula(df, "G/7.45-BRN")
        self.assertEqual(res.iloc[0][0], pytest.approx(35.598, abs=0.01))
        self.assertEqual(res.iloc[-1][0], pytest.approx(38.859, abs=0.01))


if __name__ == "__main__":
    unittest.main()
