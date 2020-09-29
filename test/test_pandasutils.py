import unittest
import pandas as pd
import cufflinks as cf
from commodutil import pandasutil


class TestPandasUtils(unittest.TestCase):

    def test_mergets(self):
        left = cf.datagen.lines(2,1000)
        right = cf.datagen.lines(2, 1000)

        res = pandasutil.mergets(left, right, leftl='Test1', rightl='Test2')
        self.assertIn('Test1', res.columns)
        self.assertIn('Test2', res.columns)
        # self.assertEqual(seas.iloc[0, -1], df[last_date.year].head(1).iloc[0][0])

    def test_sql_insert(self):
        df = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], columns=['a', 'b', 'c'])
        res = pandasutil.sql_insert_statement_from_dataframe(df, 'table')
        exp = 'INSERT INTO table (a, b, c) VALUES (1, 2, 3)'
        self.assertEqual(res[0], exp)


if __name__ == '__main__':
    unittest.main()


