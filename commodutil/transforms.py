from datetime import datetime, timedelta
from functools import reduce

import dask
from dask import delayed
import pandas as pd
import numpy as np

from commodutil import dates
from commodutil import pandasutil


def seasonailse(df, fillna=True):
    """
    Create a seasonalized DataFrame by grouping data by (month, day, year),
    averaging, and aligning to the current year.
    """
    # Ensure df is a Series if DataFrame
    if isinstance(df, pd.DataFrame):
        df = df[df.columns[0]]
    assert isinstance(df, pd.Series)

    # Remove leap day (Feb 29) entries
    s = df[~((df.index.month == 2) & (df.index.day == 29))]

    # Group by (month, day, year) to create seasonal mean
    seas = (
        s.groupby([s.index.month, s.index.day, s.index.year])
        .mean()
        .unstack()  # Unstack the year level
    )

    # seas.index is now (month, day) tuples. We convert to current year dates:
    months = [i[0] for i in seas.index]
    days = [i[1] for i in seas.index]
    newind = pd.to_datetime(
        {"year": [dates.curyear]*len(seas.index), "month": months, "day": days}
    )
    seas.index = newind

    if fillna:
        seas = pandasutil.fillna_downbet(seas)

    return seas


def seasonalize(data, histfreq: str | None = None, fillna: bool = True):
    """
    Canonical seasonalization helper.

    This mirrors the logic historically living in `commodplot.commodplottransform.seasonalise`,
    but keeps the core transformation in `commodutil`.

    Args:
        data: Series or DataFrame with a DatetimeIndex.
        histfreq: Optional frequency hint. If None, inferred from index; defaults to "D".
        fillna: Passed through to `seasonailse` for daily/monthly paths.

    Returns:
        Seasonalized DataFrame aligned to the current year index, with year columns.
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    if histfreq is None:
        histfreq = pd.infer_freq(data.index)
        if histfreq is None:
            histfreq = "D"

    if histfreq.startswith("W"):
        seas = seasonalise_weekly(data)
    else:
        # `seasonailse` expects a Series (takes first column when given DataFrame),
        # but we normalize above to DataFrame so this stays consistent with legacy behavior.
        seas = seasonailse(data, fillna=fillna)

    seas = seas.dropna(how="all", axis=1)
    return seas


def cleanup_weekly_data(df):
    """
    Processes dates in a DataFrame to ensure that the intended weekday data is present for each week.
    Fills missing weeks by carrying forward the last record of those weeks adjusted to the intended weekday.
    """

    intended_week_day = int(df['day'].mode()[0])


    # Find all possible year-week combinations and those that have the intended weekday
    all_weeks = set(pd.MultiIndex.from_product([df['year'].unique(), range(1, 53)]))
    actual_intended = df[df['day'] == intended_week_day]
    weeks_with_intended_day = set(zip(actual_intended['year'], actual_intended['week']))
    missing_weeks = all_weeks - weeks_with_intended_day

    new_records = []
    drop_indices = []

    for week in missing_weeks:
        week_mask = (df['year'] == week[0]) & (df['week'] == week[1])
        week_records = df[week_mask]
        if not week_records.empty:
            last_record = week_records.tail(1).copy()
            offset_days = int(intended_week_day - (last_record.index[0].dayofweek + 1))
            last_record.index = last_record.index + pd.Timedelta(days=offset_days)

            last_record['day'] = int(intended_week_day)


            drop_indices.extend(week_records.index)
            new_records.append(last_record)

    if drop_indices:
        df = df.drop(drop_indices)
    if new_records:
        df = pd.concat([df] + new_records)

    df = df[df['day'] == intended_week_day].sort_index()
    return df


def seasonalise_weekly(df):
    """
    Seasonalize weekly data. Similar to seasonalise but tailored for weekly frequency.
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    # Add ISO calendar columns
    df = pd.merge(df, df.index.isocalendar(), left_index=True, right_index=True)
    df = cleanup_weekly_data(df)

    # Group by year/week/dayofweek to get the first entry
    grouped = df.groupby([df.year, df.week, df.index.dayofweek]).first()

    # We now unstack to form a multi-level column structure
    df_unstacked = grouped[[df.columns[0]]].unstack().unstack()

    # Adjust columns to current year
    # Day of week is fixed from the data. Just pick the last day's info:

    # Convert to int to avoid uint32 overflow issues
    dayofweek = int(grouped['day'].iloc[-1])

    df_unstacked.columns = df_unstacked.columns.set_levels([dates.curyear], level=0)
    df_unstacked.columns = df_unstacked.columns.set_levels([dayofweek], level=1)

    # Drop week 53 if current year doesn't have it
    last_day_of_year = datetime(dates.curyear, 12, 31)
    # Condition from original code:
    # If year is leap year or not, determine if week 53 exists:
    # Simplified logic: if final day of year doesn't result in a partial week?
    # We'll trust original condition and ensure no exceptions:
    # Note: This code is somewhat unclear from original, but keep as is.
    if not (last_day_of_year.isocalendar()[1] == 53):
        # If no 53rd week, drop columns with week=53
        df_unstacked = df_unstacked.loc[:, df_unstacked.columns.get_level_values(2) != 53]

    # Convert columns to datetime from ISO format (year, dayofweek, week)
    # columns: (year, dayofweek, week)
    # map to datetime using fromisocalendar
    df_unstacked.columns = df_unstacked.columns.map(
        lambda x: datetime.fromisocalendar(x[0], x[2], x[1])
    )
    # convert int32 into int for index
    df_unstacked.index = df_unstacked.index.astype(int)

    df_result = df_unstacked.T
    return df_result


def forward_only(df):
    """
    Only take forward timeseries from the current month onwards (discarding the history).
    """
    return df[dates.curmonyear_str:]


def format_fwd(df, last_index=None):
    """
    Format a monthly-frequency forward curve into a daily series:
    - Resample daily
    - Forward fill missing values
    - If last_index is provided, truncate from that date onwards
    """
    df = df.resample("D").mean().ffill()
    if last_index is not None:

        df = df.loc[last_index:]

    return df


def _reindex_col(df, colname, colyearmap):
    """
    Reindex a single column to the current year by applying a year offset.
    Vectorized date shifting is used instead of a Python loop.
    """
    # Use .values.all() to handle both Series and DataFrame (MultiIndex columns)
    if df[colname].isnull().values.all():
        return None  # Return None instead of no return for clearer logic

    colyear = colyearmap[colname]
    delta = dates.curyear - colyear
    w = df[[colname]]

    if delta == 0:
        return w
    else:
        w = w.copy()
        w.index = w.index + pd.DateOffset(years=delta)
        return w


def reindex_year(df):
    """
    Reindex a dataframe containing prices to the current year.
    e.g. for a df with columns representing e.g. Brent Jan19, Jan18, Jan17,
    shift each past year's column forward by the appropriate number of years.
    """
    colyearmap = dates.find_year(df)
    dfs = []

    # Build a list of delayed tasks for parallelism
    for colname in df.columns:
        dfs.append(delayed(_reindex_col)(df, colname, colyearmap))

    # Compute all delayed tasks
    dfs = dask.compute(*dfs)
    dfs = [x for x in dfs if x is not None]

    # Merge all series into one dataframe via reduce and pd.merge
    if dfs:
        res = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ),
            dfs,
        )
        # Drop rows with all NaNs and fill
        res = res.dropna(how="all")
        res = pandasutil.fillna_downbet(res)
        return res
    else:
        # If all were None, return an empty DataFrame
        return pd.DataFrame(index=df.index)


def monthly_mean(df):
    """
    Given a price series, calculate the monthly mean and return as columns of means over years:
            1   2   3  ... 12
    2000    x   x   x  ... x
    2001    x   x   x  ... x

    Group by month-start and mean over that month, then pivot by month/year.
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
