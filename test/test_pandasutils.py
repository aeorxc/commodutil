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


if __name__ == '__main__':
    unittest.main()


