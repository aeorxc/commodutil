"""
Utility for forward contracts
"""


import pandas as pd

"""
Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
Return a dataframe of quarterly values (eg Brent Q115)
"""
def quarterly_contracts(c):
    years = list(set([x.year for x in c.columns]))

    dfs = []
    for year in years:
        c1, c2, c3 = '{}-01-01'.format(year), '{}-02-01'.format(year), '{}-03-01'.format(year)
        if c1 in c.columns and c2 in c.columns and c3 in c.columns:
            s = pd.concat( [c[c1], c[c2], c[c3]], 1).mean(axis=1)
            s.name = 'Q1 {}'.format(year)
            dfs.append(s)

        c4, c5, c6 = '{}-04-01'.format(year), '{}-05-01'.format(year), '{}-06-01'.format(year)
        if c4 in c.columns and c5 in c.columns and c6 in c.columns:
            s = pd.concat( [c[c3], c[c4], c[c5]], 1).mean(axis=1)
            s.name = 'Q2 {}'.format(year)
            dfs.append(s)

        c7, c8, c9 = '{}-07-01'.format(year), '{}-08-01'.format(year), '{}-09-01'.format(year)
        if c7 in c.columns and c8 in c.columns and c9 in c.columns:
            s = pd.concat( [c[c7], c[c8], c[c9]], 1).mean(axis=1)
            s.name = 'Q3 {}'.format(year)
            dfs.append(s)

        c10, c11, c12 = '{}-10-01'.format(year), '{}-11-01'.format(year), '{}-12-01'.format(year)
        if c10 in c.columns and c11 in c.columns and c12 in c.columns:
            s = pd.concat( [c[c10], c[c11], c[c12]], 1).mean(axis=1)
            s.name = 'Q4 {}'.format(year)
            dfs.append(s)

    res = pd.concat(dfs, 1)
    return res