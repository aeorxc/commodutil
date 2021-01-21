import unittest, os
import pandas as pd
from datetime import datetime
from commodutil import forwards
from commodutil import transforms
from commodutil import dates


class TestTransforms(unittest.TestCase):

    def test_seasonalise2(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(os.path.join(dirname, 'test_cl.csv'), index_col=0, parse_dates=True, dayfirst=True)

        seas = transforms.seasonailse(cl['CL_2020J'], fillna=True)
        self.assertEqual(seas[2020].loc[pd.to_datetime('%s-03-13' % dates.curyear)], 31.73)
        self.assertEqual(seas[2020].loc[pd.to_datetime('%s-03-16' % dates.curyear)], 28.7)
        self.assertEqual(seas[2020].loc[pd.to_datetime('%s-03-17' % dates.curyear)], 26.95)
        self.assertEqual(seas[2020].loc[pd.to_datetime('%s-03-18' % dates.curyear)], 20.37)
        self.assertEqual(seas[2020].loc[pd.to_datetime('%s-03-19' % dates.curyear)], 25.22)

    def test_seasonalise_weekly(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(os.path.join(dirname, 'test_weekly.csv'), index_col=0, parse_dates=True, dayfirst=True)

        seas = transforms.seasonalise_weekly(cl['PET.WCRSTUS1.W'])

        self.assertEqual(seas[2020].loc[datetime.fromisocalendar(dates.curyear, pd.to_datetime('2020-01-03').isocalendar()[1], 1)], 1066027)
        self.assertEqual(seas[2020].loc[datetime.fromisocalendar(dates.curyear, pd.to_datetime('2020-01-10').isocalendar()[1], 1)], 1063478)
        self.assertEqual(seas[2000].loc[datetime.fromisocalendar(dates.curyear, pd.to_datetime('2000-01-07').isocalendar()[1], 1)], 844791)
        self.assertEqual(seas[2000].loc[datetime.fromisocalendar(dates.curyear, pd.to_datetime('2000-12-29').isocalendar()[1], 1)], 813959)

    def test_reindex_year(self):
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
        self.assertAlmostEqual(-1.04, res['CAL 2020-2021']['%s-01-02' % (dates.curyear - 1)], 2)
        self.assertAlmostEqual(2.32, res['CAL 2021-2022']['%s-01-02' % (dates.curyear - 1)], 2)

    def test_monthly_mean(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(dirname, 'test_cl.csv'), index_col=0, parse_dates=True, dayfirst=True)[['CL_2020F']].dropna()

        res = transforms.monthly_mean(df)
        self.assertAlmostEqual(res['CL_2020F'][2015][1], 66.77, 2)


if __name__ == '__main__':
    unittest.main()


