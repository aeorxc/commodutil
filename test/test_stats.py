from commodutil import stats
from commodutil import transforms
from commodutil import forwards
import cufflinks as cf
import unittest
import os
import pandas as pd


class TestForwards(unittest.TestCase):

    def test_curve_zscore(self):
        df = cf.datagen.lines(1, 5000)
        hist = df[:'2020']
        fwd = df.resample('MS').mean()['2020':]

        res = stats.curve_seasonal_zscore(hist, fwd)

        # indendent calc
        d = stats.transforms.monthly_mean(hist).T.describe()
        z = (d[1].loc['mean'] - fwd.iloc[0][0]) / d[1].loc['std']

        self.assertAlmostEqual(res.iloc[0]['zscore'], z, 2)

    def test_reindex_zscore(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(os.path.join(dirname, 'test_cl.csv'), index_col=0, parse_dates=True, dayfirst=True)
        contracts = cl.rename(columns={x: pd.to_datetime(forwards.convert_contract_to_date(x)) for x in cl.columns})

        q = forwards.quarterly_contracts(contracts)
        q = q[[x for x in q.columns if 'Q1' in x]]

        res = stats.reindex_zscore(q)
        self.assertIsNotNone(res)


if __name__ == '__main__':
    unittest.main()