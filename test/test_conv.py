from commodutil import convfactors
import unittest
import pandas as pd


class TestUtils(unittest.TestCase):

    def test_conv_factor(self):
        res = convfactors.convfactor('diesel', 'kt', 'km3')
        self.assertAlmostEqual(res, 1.186, 3)

        res = convfactors.convfactor('ulsd', 'kt', 'km3')
        self.assertAlmostEqual(res, 1.186, 3)

        res = convfactors.convfactor('ulsd', 'km3', 'kt')
        self.assertAlmostEqual(res, 0.843, 3)

    def test_convert(self):
        diesel_kt = 520

        res = convfactors.convert(diesel_kt, 'diesel')
        self.assertAlmostEqual(res, 520, 3)

        res = convfactors.convert(diesel_kt, 'diesel', 'kt', 'km3')
        self.assertAlmostEqual(res, 616.84, 2)

    def test_convert_2(self):
        d = pd.Series([50, 60, 70, 80], index=pd.date_range('2020', periods=4, freq='MS'))
        res = convfactors.convert(d, 'diesel', 'kt', 'bbl/d')
        self.assertAlmostEqual(res['2020-1'].iloc[0], 12.02, 2)

        res = convfactors.convert(d, 'diesel', 'bbl/d', 'kt')
        self.assertAlmostEqual(res['2020-1'].iloc[0], 208.05, 2)

    def test_ec(self):

        diesel_kt = 1
        res = convfactors.convert(diesel_kt, 'diesel', 'kt', 'gj')
        self.assertAlmostEqual(res, 42.7, 2)


if __name__ == '__main__':
    unittest.main()