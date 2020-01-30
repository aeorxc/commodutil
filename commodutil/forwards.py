"""
Utility for forward contracts
"""

import pandas as pd


def quarterly_contracts(c):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of quarterly values (eg Q115)
    """
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
    # sort columns by years
    cols = list(res.columns)
    cols.sort(key=lambda s: s.split()[1])
    res = res[cols]
    return res


def quarterly_spreads(q):
    """
    Given a dataframe of quarterly contract values (eg Brent Q115, Brent Q215, Brent Q315)
    with columns headings as 'Q1 2015', 'Q2 2015'
    Return a dataframe of quarterly spreads (eg Q1-Q2 15)
    Does Q1-Q2, Q2-Q3, Q3-Q4, Q4-Q1
    """
    sprmap = {
        'Q1': 'Q2 {}',
        'Q2': 'Q3 {}',
        'Q3': 'Q4 {}',
        'Q4': 'Q1 {}',
    }

    qtrspr = []
    for col in q.columns:
        colqx = col.split(' ')[0]
        colqxyr = col.split(' ')[1]
        if colqxyr == 'Q4':
            colqxyr = int(colqxyr) + 1
        colqy = sprmap.get(colqx).format(colqxyr)
        if colqy in q.columns:
            r = q[col] - q[colqy]
            r.name = '{}-{} {}'.format(colqx, colqy.split(' ')[0], colqxyr)
            qtrspr.append(r)

    res = pd.concat([qtrspr], 1)
    return res