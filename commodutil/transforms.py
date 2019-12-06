import pandas as pd
import re
from functools import reduce
from commodutil import dates


def seasonailse(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    assert isinstance(df, pd.DataFrame)
    if len(df.columns) > 1:
        df = df[[df.columns[0]]]

    s = df[~((df.index.month == 2) & (df.index.day == 29))]  # remove leap dates 29 Feb
    seas = s.groupby([s.index.month, s.index.day, s.index.year, ]).mean().unstack()

    # replace index with dates from current year
    newind = [pd.to_datetime('{}-{}-{}'.format(dates.curyear, i[0], i[1])) for i in seas.index]
    seas.index = newind
    seas = seas[df.columns[0]] # remove multi-index created from group by

    return seas


"""
Only take forward timeseries from cur month onwards (discarding the history)
"""
def forward_only(df):
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


"""
Reindex a dataframe containing prices to the current year.
eg dataframe with brent Jan 19, Jan 18, Jan 17   so that 18 is shifted +1 year and 17 is shifted +2 years 
"""
def reindex_year(df):
    dfs = []
    for colname in df.columns:
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
    res = res.fillna(method='ffill', limit=4) # fill weekends

    return res


if __name__ == '__main__':
    pass