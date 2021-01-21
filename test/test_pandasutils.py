import unittest
import pandas as pd
from commodutil import pandasutil
from commodutil import forwards
import os


class TestPandasUtils(unittest.TestCase):

    def test_mergets(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(os.path.join(dirname, 'test_cl.csv'), index_col=0, parse_dates=True, dayfirst=True)
        contracts = cl.rename(columns={x: pd.to_datetime(forwards.convert_contract_to_date(x)) for x in cl.columns})

        res = pandasutil.mergets(contracts['2020-01-01'], contracts['2020-02-01'], leftl='Test1', rightl='Test2')
        self.assertIn('Test1', res.columns)
        self.assertIn('Test2', res.columns)

    def test_sql_insert(self):
        df = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], columns=['a', 'b', 'c'])
        res = pandasutil.sql_insert_statement_from_dataframe(df, 'table')
        exp = 'INSERT INTO table (a, b, c) VALUES (1, 2, 3)'
        self.assertEqual(res[0], exp)


if __name__ == '__main__':
    unittest.main()


