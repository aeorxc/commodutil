import os
import unittest

import pandas as pd

from commodutil import forwards


class TestForwards(unittest.TestCase):

    def test_timespreads(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )
        contracts = cl

        res = forwards.time_spreads(contracts, m1=6, m2=12)
        self.assertAlmostEqual(res[2019].loc[pd.to_datetime("2019-01-02")], -1.51, 2)
        self.assertAlmostEqual(res[2019].loc[pd.to_datetime("2019-05-21")], 0.37, 2)

        res = forwards.time_spreads(contracts, m1=12, m2=12)
        self.assertAlmostEqual(res[2019].loc[pd.to_datetime("2019-11-20")], 3.56, 2)
        self.assertAlmostEqual(res[2020].loc[pd.to_datetime("2019-03-20")], 2.11, 2)

    def test_timespreads2(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.time_spreads(cl, m1="Q1", m2="Q2")
        self.assertAlmostEqual(res[2020].loc[pd.to_datetime("2019-01-02")], -0.33, 2)
        self.assertAlmostEqual(res[2020].loc[pd.to_datetime("2019-05-21")], 1.05, 2)

        res = forwards.time_spreads(cl, m1="Q4", m2="Q1")
        self.assertAlmostEqual(res[2020].loc[pd.to_datetime("2019-01-02")], -0.25, 2)
        self.assertAlmostEqual(res[2020].loc[pd.to_datetime("2019-05-21")], 0.91, 2)

    def test_spread_combinations(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combinations(cl)
        self.assertIn("Q1", res)
        self.assertIn("Q1Q2", res)
        self.assertIn("Calendar", res)
        self.assertIn("JanFeb", res)
        self.assertIn("JanFebMar", res)

    def test_spread_combination_calendar(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "calendar")
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res[2020]["2020-01-02"], 59.174, 3)

    def test_spread_combination_calendar_spread(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "calendar spread")
        self.assertAlmostEqual(res["CAL 2020-2021"]["2020-01-02"], 4.35, 2)

    def test_spread_combination_half_year(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "half year")
        self.assertAlmostEqual(res["H1 2020"]["2019-01-02"], 50.04, 2)

    def test_spread_combination_half_year_spread(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "half year spread")
        self.assertAlmostEqual(res["H1H2 2020"]["2019-01-02"], -0.578, 2)

    def test_spread_combination_quarter(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )
        contracts = cl

        res = forwards.spread_combination(contracts, "q1")
        self.assertAlmostEqual(res["Q1 2020"]["2019-01-02"], 49.88, 2)

    def test_spread_combination_quarter_spread(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "q1q2")
        self.assertAlmostEqual(res["Q1Q2 2020"]["2019-01-02"], -0.33, 2)

        res = forwards.spread_combination(cl, "q1q3")
        self.assertAlmostEqual(res["Q1Q3 2020"]["2019-01-02"], -0.58, 2)

    def test_spread_combination_month(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "jan")
        self.assertAlmostEqual(res["Jan 2020"]["2019-01-02"], 49.77, 2)

    def test_spread_combination_month_spread_janfeb(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "janfeb")
        self.assertAlmostEqual(res["JanFeb 2020"]["2019-01-02"], -0.11, 2)

    def test_spread_combination_month_spread_decjan(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "decjan")
        self.assertAlmostEqual(res["DecJan 2020"]["2019-01-02"], -0.06, 2)

    def test_spread_combination_month_fly(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "janfebmar")
        self.assertAlmostEqual(res["JanFebMar 2020"]["2019-01-02"], 0.0, 2)

    def test_spread_combination_quarter_fly(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        res = forwards.spread_combination(cl, "q4q1q2")
        self.assertAlmostEqual(res["Q4Q1Q2 2020"]["2019-01-02"], -0.023, 3)

    def test_continuous_futures(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        cl = pd.read_csv(
            os.path.join(dirname, "test_cl.csv"),
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        )

        cl = cl.rename(
            columns={
                x: pd.to_datetime(forwards.convert_contract_to_date(x))
                for x in cl.columns
            }
        )

        expiry_dates = {
            "2019-01-01": "2018-12-19",
            "2019-02-01": "2019-01-22",
            "2019-03-01": "2019-02-20",
            "2019-04-01": "2019-03-20",
            "2019-05-01": "2019-04-22",
            "2019-06-01": "2019-05-21",
            "2019-07-01": "2019-06-20",
            "2019-08-01": "2019-07-22",
            "2019-09-01": "2019-08-20",
            "2019-10-01": "2019-09-20",
            "2019-11-01": "2019-10-22",
            "2019-12-01": "2019-11-20",
            "2020-01-01": "2019-12-19",
            "2020-02-01": "2020-01-21",
            "2020-03-01": "2020-02-20",
            "2020-04-01": "2020-03-20",
            "2020-05-01": "2020-04-21",
            "2020-06-01": "2020-05-19",
            "2020-07-01": "2020-06-22",
            "2020-08-01": "2020-07-21",
            "2020-09-01": "2020-08-20",
            "2020-10-01": "2020-09-22",
            "2020-11-01": "2020-10-20",
            "2020-12-01": "2020-11-20",
            "2021-01-01": "2021-01-20",
        }

        res = forwards.continuous_futures(cl, expiry_dates=expiry_dates, front_month=[1,2])
        self.assertAlmostEqual(res["M1"]["2020-11-20"], 42.15, 3)
        self.assertAlmostEqual(res["M1"]["2020-11-23"], 43.06, 3)
        self.assertAlmostEqual(res["M2"]["2020-11-19"], 41.90, 3)
        self.assertAlmostEqual(res["M2"]["2020-11-20"], 42.42, 3)
        self.assertAlmostEqual(res["M2"]["2020-11-23"], 43.28, 3)

        res = forwards.continuous_futures(cl, expiry_dates=expiry_dates, roll_days=1)
        self.assertAlmostEqual(res["M1"]["2020-11-19"], 41.74, 3)
        self.assertAlmostEqual(res["M1"]["2020-11-20"], 42.42, 3)
        self.assertAlmostEqual(res["M1"]["2020-11-23"], 43.06, 3)


        res = forwards.continuous_futures(
            cl, expiry_dates=expiry_dates, front_month=2, roll_days=1
        )
        self.assertAlmostEqual(res["M2"]["2020-11-19"], 41.90, 3)
        self.assertAlmostEqual(res["M2"]["2020-11-20"], 42.64, 3)
        self.assertAlmostEqual(res["M2"]["2020-11-23"], 43.28, 3)

        res = forwards.continuous_futures(
            cl, expiry_dates=expiry_dates, front_month=1, back_adjust=True
        )
        self.assertAlmostEqual(res["M1"]["2020-11-19"], 42.01, 3)
        self.assertAlmostEqual(res["M1"]["2020-11-20"], 42.42, 3)
        self.assertAlmostEqual(res["M1"]["2020-11-23"], 43.06, 3)


if __name__ == "__main__":
    unittest.main()
