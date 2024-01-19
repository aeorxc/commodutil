from datetime import datetime, timedelta
from functools import reduce

import dask
import pandas as pd

from commodutil import dates
from commodutil import pandasutil


def seasonailse(df, fillna=True):
    if isinstance(df, pd.DataFrame):
        df = pd.Series(df[df.columns[0]])

    assert isinstance(df, pd.Series)

    s = df[~((df.index.month == 2) & (df.index.day == 29))]  # remove leap dates 29 Feb
    seas = (
        s.groupby(
            [
                s.index.month,
                s.index.day,
                s.index.year,
            ]
        )
        .mean()
        .unstack()
    )

    # replace index with dates from current year
    newind = [
        pd.to_datetime("{}-{}-{}".format(dates.curyear, i[0], i[1])) for i in seas.index
    ]
    seas.index = newind

    if fillna:
        seas = pandasutil.fillna_downbet(seas)

    return seas


def cleanup_weekly_data(df):
    """
    Processes the dates in a DataFrame according to the specified rules and fills missing weeks.

    :param df: Pandas DataFrame to be processed.
    :return: processed DataFrame.
    """
    intended_week_day = df['day'].mode()[0]
    all_weeks = set(pd.MultiIndex.from_product([df['year'].unique(), range(1, 53)]))
    weeks_with_intended_day = set(zip(df[df['day'] == intended_week_day]['year'], df[df['day'] == intended_week_day]['week']))
    missing_weeks = all_weeks - weeks_with_intended_day

    for week in missing_weeks:
        week_records = df[(df['year'] == week[0]) & (df['week'] == week[1])]
        if not week_records.empty:
            last_record = week_records.tail(1)
            last_record.index = last_record.index.map(lambda x: x + timedelta(days=int(intended_week_day - (x.dayofweek+1))))
            last_record.day = intended_week_day
            df = df.drop(week_records.index)
            df = df.append(last_record)

    df = df[df['day'] == intended_week_day]
    return df.sort_index()


def seasonalise_weekly(df):
    """
    Edge case for handling weekly data - eg DOE where we need to tweak the standard
    seasonalise() method.
    :param df:
    :return:
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    df = pd.merge(df, df.index.isocalendar(), left_index=True, right_index=True)
    df = cleanup_weekly_data(df)
    df = df.groupby([df.year, df.week, df.index.dayofweek]).first()
    dayofweek = df['day'].iloc[-1]

    df = df[[df.columns[0]]].unstack().unstack()
    first_level = df.columns.levels[0][0]  # Get the first level of the MultiIndex
    df.columns = df.columns.set_levels([dates.curyear for x in df.columns.levels[0]], level=0)
    df.columns = df.columns.set_levels([dayofweek for x in df.columns.levels[0]], level=1)

    # drop week 53 if the current year has no week 53
    last_day_of_year = datetime(dates.curyear, 12, 31)
    if not last_day_of_year.weekday() >= 3 if dates.curyear % 4 == 0 else last_day_of_year.weekday() >= 2:
        df = df.loc[:, df.columns.get_level_values(2) != 53]

    # convert the columns to datetime
    df.columns = df.columns.map(lambda x: datetime.fromisocalendar(x[0], x[2], x[1]))
    df = df.T
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
    df = df.resample("D").mean().fillna(method="ffill")
    if last_index is not None:
        df = df[last_index:]

    return df


def _reindex_col(df, colname, colyearmap):
    if df[colname].isnull().all():
        return  # logic below wont work on all empty NaN columns

    # determine year
    colyear = colyearmap[colname]
    delta = dates.curyear - colyear
    w = df[[colname]]
    if delta == 0:
        return w
    else:  # reindex
        winew = [x + pd.DateOffset(years=delta) for x in w.index]
        w.index = winew
        return w


def reindex_year(df):
    """
    Reindex a dataframe containing prices to the current year.
    eg dataframe with brent Jan 19, Jan 18, Jan 17   so that 18 is shifted +1 year and 17 is shifted +2 years
    """
    dfs = []
    colyearmap = dates.find_year(df)
    for colname in df.columns:
        dfs.append(dask.delayed(_reindex_col(df, colname, colyearmap)))

    dfs = dask.compute(*dfs)
    dfs = [x for x in dfs if x is not None]
    # merge all series into one dataframe, concat doesn't quite do the job
    res = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="outer"
        ),
        dfs,
    )
    res = res.dropna(how="all")  # drop uneeded columns out into future
    res = pandasutil.fillna_downbet(
        res
    )  # use this as above ffills incorrectly at end of timeseries

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
    monthly_mean = df.groupby(pd.Grouper(freq="MS")).mean()
    month_pivot = (
        monthly_mean.groupby([monthly_mean.index.month, monthly_mean.index.year])
        .sum()
        .unstack()
    )
    return month_pivot


if __name__ == "__main__":
    pass
