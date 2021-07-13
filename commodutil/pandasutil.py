import pandas as pd
import numpy as np


def mergets(left, right, leftl=None, rightl=None, how='left'):
    """
    Wrapper for pandas merge for working on merging timeseries
    """

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
    for col in df.columns:
        non_nans = df[col][~df[col].apply(np.isnan)]
        if non_nans is not None and len(non_nans) > 1:
            start, end = non_nans.index[0], non_nans.index[-1]
            df[col].loc[start:end] = df[col].loc[start:end].fillna(method='ffill')
    return df


def sql_insert_statement_from_dataframe(df, table_name):
    """
    Turn a dataframe into a set of insert statements
    Taken from https://stackoverflow.com/questions/31071952/generate-sql-statements-from-a-pandas-dataframe
    :param df:
    :param table_name:
    :return:
    """
    sql_texts = []
    for index, row in df.iterrows():
        q = 'INSERT INTO ' + table_name + ' (' + str(', '.join(df.columns)) + ') VALUES ' + str(tuple(row.values))
        q = q.replace('nan', 'Null').replace('None', 'Null')
        print(q)
        sql_texts.append(q)

    return sql_texts
