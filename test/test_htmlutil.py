import unittest
from commodutil import htmlutil


class TestHtmlUtil(unittest.TestCase):

    def test_getbasehtml(self):
        st = htmlutil.getbasehtml('Test Title')
        self.assertIn('Test Title', st)
        self.assertIn(htmlutil.bast_style, st)


if __name__ == '__main__':
    unittest.main()


