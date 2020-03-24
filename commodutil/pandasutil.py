import pandas as pd
import numpy as np

"""
Wrapper for pandas merge for working on merging timeseries
"""
def mergets(left, right, leftl=None, rightl=None, how='left'):

    if isinstance(left, pd.Series):
        left = pd.DataFrame(left)
    if isinstance(right, pd.Series):
        right = pd.DataFrame(right)
    res = pd.merge(left, right, left_index=True, right_index=True, how=how)

    rename = {}
    if leftl is not None:
        rename[left.columns[0]] = leftl
        rename['{}_x'.format(left.columns[0])] = leftl
    if rightl is not None:
        rename[right.columns[0]] = rightl
        rename['{}_y'.format(right.columns[0])] = rightl

    res = res.rename(columns=rename)

    return res


def fillna_downbet(df):
    """
    Fill weekends/holidays in timeseries but dont extend to NaNs at end of the timeseries
    https://stackoverflow.com/questions/28136663/using-pandas-to-fill-gaps-only-and-not-nans-on-the-ends
    :param df:
    :return:
    """
    df = df.copy()
    for col in df:
        non_nans = df[col][~df[col].apply(np.isnan)]
        start, end = non_nans.index[0], non_nans.index[-1]
        df[col].loc[start:end] = df[col].loc[start:end].fillna(method='ffill')
    return df