import pandas as pd
import re
from functools import reduce
from commodutil import dates
from commodutil import pandasutil


def seasonailse(df, fillna=True):
    if isinstance(df, pd.DataFrame):
        df = pd.Series(df[df.columns[0]])

    assert isinstance(df, pd.Series)

    s = df[~((df.index.month == 2) & (df.index.day == 29))]  # remove leap dates 29 Feb
    seas = s.groupby([s.index.month, s.index.day, s.index.year, ]).mean().unstack()

    # replace index with dates from current year
    newind = [pd.to_datetime('{}-{}-{}'.format(dates.curyear, i[0], i[1])) for i in seas.index]
    seas.index = newind

    if fillna:
        seas = pandasutil.fillna_downbet(seas)

    return seas


def forward_only(df):
    """
    Only take forward timeseries from cur month onwards (discarding the history)
    """
    df = df[dates.curmonyear_str:]
    return df


"""
Format a monthly-frequency forward curve into a daily series 
"""
def format_fwd(df, last_index=None):
    df = df.resample('D').mean().fillna(method='ffill')
    if last_index is not None:
        df = df[last_index:]

    return df


def reindex_year(df):
    """
    Reindex a dataframe containing prices to the current year.
    eg dataframe with brent Jan 19, Jan 18, Jan 17   so that 18 is shifted +1 year and 17 is shifted +2 years
    """
    dfs = []
    for colname in df.columns:
        if df[colname].isnull().all():
            continue # logic below wont work on all empty NaN columns

        # determine year
        colregex = re.findall('\d\d\d\d', str(colname))
        if len(colregex) == 1:
            colyear = int(colregex[0])
            # determine offset
            delta = dates.curyear - colyear

            w = df[[colname]]
            if delta == 0:
                dfs.append(w)
            else: # reindex
                winew = [x + pd.DateOffset(years=delta) for x in w.index]
                w.index = winew
                dfs.append(w)

    # merge all series into one dataframe, concat doesn't quite do the job
    res = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'),dfs)
    res = res.dropna(how='all') # drop uneeded columns out into future
    # res = res.fillna(method='ffill', limit=4) # fill weekends
    res = pandasutil.fillna_downbet(res) # use this as above ffills incorrectly at end of timeseries

    return res


if __name__ == '__main__':
    pass