import re
from functools import reduce

import numpy as np
import pandas as pd


def mergets(left, right, leftl=None, rightl=None, how="left"):
    """
    Wrapper for pandas merge for working on merging timeseries
    """

    if isinstance(left, pd.Series):
        left = pd.DataFrame(left)
    if isinstance(right, pd.Series):
        right = pd.DataFrame(right)
    if len(left.columns) == 0:
        right["left"] = pd.Series(dtype="float64")
    if len(right.columns) == 0:
        right["right"] = pd.Series(dtype="float64")

    res = pd.merge(left, right, left_index=True, right_index=True, how=how)

    rename = {}
    if leftl is not None:
        rename[left.columns[0]] = leftl
        rename["{}_x".format(left.columns[0])] = leftl
    if rightl is not None:
        rename[right.columns[0]] = rightl
        rename["{}_y".format(right.columns[0])] = rightl

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
            df[col].loc[start:end] = df[col].loc[start:end].fillna(method="ffill")
    return df


def sql_insert_statement_from_dataframe(df, table_name, print_statemnt=False):
    """
    Turn a dataframe into a set of insert statements
    Taken from https://stackoverflow.com/questions/31071952/generate-sql-statements-from-a-pandas-dataframe
    :param df:
    :param table_name:
    :return:
    """
    sql_texts = []
    for index, row in df.iterrows():
        vals = [re.sub(r"\'", "", x) if isinstance(x, str) else x for x in row.values]
        q = (
            "INSERT INTO "
            + table_name
            + " ("
            + str(", ".join(df.columns))
            + ") VALUES "
            + str(tuple(vals))
        )
        q = q.replace("nan", "Null").replace("None", "Null")
        if print_statemnt:
            print(q)
        sql_texts.append(q)

    return sql_texts


def mergem(c):
    "Wrapper method to merge multiple data frames"
    c = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="outer"
        ),
        c,
    )
    return c


def generate_lambda(expression, columns):
    """
    Convert a simple expression string into a lambda function suitable for DataFrame operations.
    """
    if isinstance(columns, pd.MultiIndex):
        columns = columns.levels[0]

    # Sort columns by length in descending order to handle substring issues
    columns = sorted(columns, key=len, reverse=True)

    for col in columns:
        # Use regular expressions to replace whole word matches
        expression = re.sub(rf"\b{re.escape(col)}\b", f"x['{col}']", expression)

    return lambda x: eval(expression)


def apply_formula(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Given a dataframe apply the formula to the columns in the dataframe
    """

    formula_l = generate_lambda(formula, df.columns)
    dfr = df.apply(formula_l, axis=1)

    if isinstance(dfr, pd.Series):
        dfr = pd.DataFrame(dfr, columns=[formula])

    return dfr
