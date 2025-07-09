import unittest

import pandas as pd
from commodutil import arb

import pytest


class TestArb(unittest.TestCase):

    def test_arb_calc(self):
        data = {
            'RB': [967.257942, 973.695366, 981.147384, 995.386686, 1024.530024],
            'EBOB': [939.75, 944.75, 947.75, 940.50, 965.00],
            'TC2': [29.031, 28.807, 29.182, 29.021, 29.592]
        }

        index_dates = ['2023-08-03', '2023-08-04', '2023-08-07', '2023-08-08', '2023-08-09']
        df = pd.DataFrame(data, index=index_dates)
        res = arb.calc(df, 'RB', 'EBOB', 'TC2')
        self.assertEqual(res.iloc[0][0], pytest.approx(-1.52, abs=0.01))


if __name__ == "__main__":
    unittest.main()
