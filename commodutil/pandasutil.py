import pandas as pd


"""
Wrapper for pandas merge for working on merging timeseries
"""
def mergets(left, right, leftl=None, rightl=None, how='left'):

    res = pd.merge(left, right, left_index=True, right_index=True, how=how)

    rename = {}
    if leftl is not None:
        rename[left.columns[0]] = leftl
    if rightl is not None:
        rename[right.columns[0]] = rightl

    res = res.rename(columns=rename)

    return res