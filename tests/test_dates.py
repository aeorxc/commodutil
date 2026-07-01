import unittest

import pandas as pd

from commodutil import dates


class TestDates(unittest.TestCase):
    def test_find_year(self):
        df = pd.DataFrame(columns=["Q1 2020", "Q2 2022"])
        res = dates.find_year(df)
        self.assertEqual(res["Q1 2020"], 2020)
        self.assertEqual(res["Q2 2022"], 2022)

    def test_find_year2(self):
        df = pd.DataFrame(columns=["CAL 2020-2021"])
        res = dates.find_year(df)
        self.assertEqual(res["CAL 2020-2021"], 2020)

    def test_find_year3(self):
        df = pd.DataFrame(columns=["FB", "FP"])
        res = dates.find_year(df)
        self.assertEqual(res["FB"], "FB")
        self.assertEqual(res["FP"], "FP")

    def test_find_year4(self):
        df = pd.DataFrame(columns=["FB", "FP 2021"])
        res = dates.find_year(df)
        self.assertEqual(res["FB"], "FB")
        self.assertEqual(res["FP 2021"], 2021)

    def test_time_until_end_of_day_runs(self):
        # regression: dates.py imported the stdlib `time` module, so `time.min`
        # raised AttributeError. It must reference datetime.time instead.
        import datetime as _dt

        secs = dates.time_until_end_of_day()
        self.assertIsInstance(secs, int)
        self.assertTrue(0 <= secs <= 86400)
        self.assertEqual(
            dates.time_until_end_of_day(_dt.datetime(2026, 1, 1, 23, 0, 0)), 3600
        )


if __name__ == "__main__":
    unittest.main()
