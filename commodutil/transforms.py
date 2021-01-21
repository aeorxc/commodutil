import pandas as pd
from datetime import datetime
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


def seasonalise_weekly(df, freq='W'):
    """
    Edge case for handling weekly data - eg DOE where we need to tweak the standard
    seasonalise() method.
    :param df:
    :return:
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    df = pd.merge(df, df.index.isocalendar(), left_index=True, right_index=True)
    df = df.groupby([df.week, df.year]).mean()[df.columns[0]].unstack()
    # when converting back to date format, some years don't have week 53 so drop for now
    if 53 in df.index and pd.to_datetime('%s-12-31' % dates.curyear).isocalendar()[1] == 52:
        df = df.drop(53)
    df.index = df.index.map(lambda x: datetime.fromisocalendar(2021, x, 1))
    return df


def forward_only(df):
    """
    Only take forward timeseries from cur month onwards (discarding the history)
    """
    df = df[dates.curmonyear_str:]
    return df


def format_fwd(df, last_index=None):
    """
    Format a monthly-frequency forward curve into a daily series
    """
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
    colyearmap = dates.find_year(df)
    for colname in df.columns:
        if df[colname].isnull().all():
            continue # logic below wont work on all empty NaN columns

        # determine year
        colyear = colyearmap[colname]
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
    res = pandasutil.fillna_downbet(res) # use this as above ffills incorrectly at end of timeseries

    return res


def monthly_mean(df):
    """
    Given a price series, calculate the monthly mean and return as columns of means over years
            1  2  3 .. 12
    2000    x  x  x .. x
    2001    x  x  x .. x

    :param df:
    :return:
    """
    monthly_mean = df.groupby(pd.Grouper(freq='MS')).mean()
    month_pivot = monthly_mean.groupby([monthly_mean.index.month, monthly_mean.index.year]).sum().unstack()
    return month_pivot


if __name__ == '__main__':
    pass