import unittest
import pandas as pd
import cufflinks as cf
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
        # self.assertEqual(seas.iloc[0, -1], df[last_date.year].head(1).iloc[0][0])

    def test_reindex_year(self):
        df = cf.datagen.lines(4, 10000)
        years = [x for x in range(dates.curyear - 2, dates.curyear + 2)]
        m = dict(zip(df.columns, years))
        df = df.rename(columns=m)

        res = transforms.reindex_year(df)

        self.assertEqual(df.loc['{}-01-01'.format(dates.curyear), dates.curyear], res.loc['{}-01-01'.format(dates.curyear), dates.curyear])
        self.assertEqual(df.loc['{}-01-01'.format(dates.curyear), dates.curyear-1], res.loc['{}-01-01'.format(dates.curyear-1), dates.curyear])
        self.assertEqual(df.loc['{}-01-01'.format(dates.curyear), dates.curyear+1], res.loc['{}-01-01'.format(dates.curyear+1), dates.curyear])


if __name__ == '__main__':
    unittest.main()


