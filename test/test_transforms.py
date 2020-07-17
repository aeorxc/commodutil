import unittest, os
import pandas as pd
import cufflinks as cf
from commodutil import forwards
from commodutil import transforms
from commodutil import dates


class TestTransforms(unittest.TestCase):

    def test_seasonalise(self):
        df = cf.datagen.lines(2,1000)
        col = df.columns[0]
        seas = transforms.seasonailse(df)

        first = df.iloc[0, 0]
        last_date = df.index[-1]
        last_val = df.tail(1)[col].iloc[0]

        self.assertEqual(seas.iloc[0,0], first)
        for col in seas.columns: # check columns are 4 digitis long ie years
            self.assertTrue(len(str(col)), 4)
        # self.assertEqual(seas.iloc[0, -1], df[last_date.year].head(1).iloc[0][0])

    def test_seasonalise2(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(os.path.join(dirname, 'test_cl.csv'), index_col=0, parse_dates=True, dayfirst=True)

        seas = transforms.seasonailse(cl['CL_2020J'], fillna=True)
        self.assertEqual(seas[2020].loc[pd.to_datetime('2020-03-15')], 31.73)
        self.assertEqual(seas[2020].loc[pd.to_datetime('2020-03-16')], 28.7)
        self.assertEqual(seas[2020].loc[pd.to_datetime('2020-03-17')], 26.95)
        self.assertEqual(seas[2020].loc[pd.to_datetime('2020-03-18')], 20.37)
        self.assertEqual(seas[2020].loc[pd.to_datetime('2020-03-19')], 25.22)

    def test_reindex_year(self):
        df = cf.datagen.lines(4, 10000)
        years = [x for x in range(dates.curyear - 2, dates.curyear + 2)]
        m = dict(zip(df.columns, years))
        df = df.rename(columns=m)

        res = transforms.reindex_year(df)

        # current year series stays the same
        self.assertEqual(df.loc['{}-01-01'.format(dates.curyear), dates.curyear], res.loc['{}-01-01'.format(dates.curyear), dates.curyear])
        # previous year moves forward by 1 year
        self.assertEqual(df.loc['{}-01-01'.format(dates.curyear-1), dates.curyear-1], res.loc['{}-01-01'.format(dates.curyear), dates.curyear-1])
        # subsequent year moves back by 1 year
        self.assertEqual(df.loc['{}-01-01'.format(dates.curyear+1), dates.curyear+1], res.loc['{}-01-01'.format(dates.curyear), dates.curyear+1])

    def test_reindex_year2(self):
        """
        Test reindex with cal spread (eg Cal 20-21, Cal 21-22)
        :return:
        """
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(os.path.join(dirname, 'test_cl.csv'), index_col=0, parse_dates=True, dayfirst=True)
        contracts = cl.rename(columns={x: pd.to_datetime(forwards.convert_contract_to_date(x)) for x in cl.columns})
        qorig = forwards.cal_contracts(contracts)
        cal_sp = forwards.cal_spreads(qorig)

        res = transforms.reindex_year(cal_sp)
        self.assertAlmostEqual(-1.04, res['CAL 2020-2021']['2019-01-02'], 2)
        self.assertAlmostEqual(2.32, res['CAL 2021-2022']['2019-01-02'], 2)

    def test_monthly_mean(self):
        df = cf.datagen.lines(4, 10000)
        res = transforms.monthly_mean(df)

        col = df.columns[0]
        month1 = df.index[0]
        mean1 = df[month1.strftime('%Y-%m')][col].mean()
        self.assertAlmostEqual(mean1, res[col][month1.year][month1.month], 2)


if __name__ == '__main__':
    unittest.main()


