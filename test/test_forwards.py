from commodutil import forwards
import unittest


class TestForwards(unittest.TestCase):

    def test_conv_factor(self):
        res = forwards.convert_contract_to_date('2020F')
        self.assertEqual(res, '2020-1-1')


if __name__ == '__main__':
    unittest.main()