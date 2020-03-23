from commodutil import forwards
import unittest
import os
import pandas as pd


class TestForwards(unittest.TestCase):

    def test_conv_factor(self):
        res = forwards.convert_contract_to_date('2020F')
        self.assertEqual(res, '2020-1-1')

    def test_quarterly_contracts(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(os.path.join(dirname, 'test_cl.csv'), index_col=0, parse_dates=True, dayfirst=True)
        contracts = cl.rename(columns={x: pd.to_datetime(forwards.convert_contract_to_date(x)) for x in cl.columns})

        res = forwards.quarterly_contracts(contracts)
        self.assertAlmostEqual(res['Q2 2019'].loc[pd.to_datetime('2019-03-20')], 60.18, 2)
        self.assertAlmostEqual(res['Q3 2019'].loc[pd.to_datetime('2019-06-20')], 56.95, 2)
        self.assertAlmostEqual(res['Q4 2019'].loc[pd.to_datetime('2019-09-20')], 58.01, 2)
        self.assertAlmostEqual(res['Q1 2020'].loc[pd.to_datetime('2019-12-19')], 61.09, 2)

        self.assertAlmostEqual(res['Q2 2020'].loc[pd.to_datetime('2020-03-20')], 23.14, 2)

        res = forwards.quarterly_spreads(res)
        self.assertAlmostEqual(res['Q1-Q2 2020'].loc[pd.to_datetime('2019-12-19')], 1.14, 2)
        self.assertAlmostEqual(res['Q2-Q3 2019'].loc[pd.to_datetime('2019-03-20')], -0.73, 2)
        self.assertAlmostEqual(res['Q3-Q4 2019'].loc[pd.to_datetime('2019-06-20')], 0.07, 2)
        self.assertAlmostEqual(res['Q4-Q1 2020'].loc[pd.to_datetime('2019-09-20')], 1.12, 2)



if __name__ == '__main__':
    unittest.main()