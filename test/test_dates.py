from commodutil import dates
import unittest
import pandas as pd


class TestDates(unittest.TestCase):

    def test_find_year(self):
        df = pd.DataFrame(columns=['Q1 2020', 'Q2 2022'])
        res = dates.find_year(df)
        self.assertEqual(res['Q1 2020'], 2020)
        self.assertEqual(res['Q2 2022'], 2022)

    def test_find_year2(self):
        df = pd.DataFrame(columns=['CAL 2020-2021'])
        res = dates.find_year(df)
        self.assertEqual(res['CAL 2020-2021'], 2020)

    def test_find_year3(self):
        df = pd.DataFrame(columns=['FB', 'FP'])
        res = dates.find_year(df)
        self.assertEqual(res['FB'], 'FB')
        self.assertEqual(res['FP'], 'FP')

    def test_find_year4(self):
        df = pd.DataFrame(columns=['FB', 'FP 2021'])
        res = dates.find_year(df)
        self.assertEqual(res['FB'], 'FB')
        self.assertEqual(res['FP 2021'], 2021)


if __name__ == '__main__':
    unittest.main()