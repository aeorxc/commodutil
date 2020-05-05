"""
Utility for forward contracts
"""
import re
import pandas as pd
from commodutil import dates

futures_month_conv = {
        1: "F",
        2: "G",
        3: "H",
        4: "J",
        5: "K",
        6: "M",
        7: "N",
        8: "Q",
        9: "U",
        10: "V",
        11: "X",
        12: "Z"
    }

futures_month_conv_inv =  {v: k for k, v in futures_month_conv.items()}


def convert_contract_to_date(contract):
    """
    Given a string like FB_2020J return 2020-01-01
    :param contract:
    :return:
    """
    c = re.findall('\d\d\d\d\w', contract)
    if len(c) > 0:
        c = c[0]
    d = '{}-{}-1'.format(c[:4], futures_month_conv_inv.get(c[-1], 0))
    return d


def time_spreads(contracts, m1, m2):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of time spreads  (eg m1 = 12, m2 = 12 gives Dec-Dec spread)
    """

    cf = [x for x in contracts if x.month == m1]
    dfs = []

    for c1 in cf:
        year1, year2 = c1.year, c1.year
        if m1 == m2:
            year2 = year1 + 1
        c2 = [x for x in contracts if x.month == m2 and x.year == year2]
        if len(c2) == 1:
            c2 = c2[0]
            s = contracts[c1] - contracts[c2]
            s.name = year1
            dfs.append(s)

    res = pd.concat(dfs, 1)
    res = res.dropna(how='all', axis=1)
    return res


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
            s = pd.concat([c[c1], c[c2], c[c3]], 1).dropna(how='any').mean(axis=1)
            s.name = 'Q1 {}'.format(year)
            dfs.append(s)

        c4, c5, c6 = '{}-04-01'.format(year), '{}-05-01'.format(year), '{}-06-01'.format(year)
        if c4 in c.columns and c5 in c.columns and c6 in c.columns:
            s = pd.concat([c[c4], c[c5], c[c6]], 1, sort=True).dropna(how='any').mean(axis=1)
            s.name = 'Q2 {}'.format(year)
            dfs.append(s)

        c7, c8, c9 = '{}-07-01'.format(year), '{}-08-01'.format(year), '{}-09-01'.format(year)
        if c7 in c.columns and c8 in c.columns and c9 in c.columns:
            s = pd.concat( [c[c7], c[c8], c[c9]], 1).dropna(how='any').mean(axis=1)
            s.name = 'Q3 {}'.format(year)
            dfs.append(s)

        c10, c11, c12 = '{}-10-01'.format(year), '{}-11-01'.format(year), '{}-12-01'.format(year)
        if c10 in c.columns and c11 in c.columns and c12 in c.columns:
            s = pd.concat( [c[c10], c[c11], c[c12]], 1).dropna(how='any').mean(axis=1)
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
        if colqx == 'Q4':
            colqxyr = int(colqxyr) + 1
        colqy = sprmap.get(colqx).format(colqxyr)
        if colqy in q.columns:
            r = q[col] - q[colqy]
            r.name = '{}-{} {}'.format(colqx, colqy.split(' ')[0], colqxyr)
            qtrspr.append(r)

    res = pd.concat(qtrspr, 1, sort=True)
    return res


def relevant_qtr_contract(qx):
    """
    Given a qtr, eg, Q1, determine the right year to use in seasonal charts.
    For example after Feb 2020, use Q1 2021 as Q1 2020 would have stopped pricing

    :param qx:
    :return:
    """
    relyear = dates.curyear
    if qx == 'Q1':
        if dates.curmon >= 1:
            relyear = relyear + 1
    elif qx == 'Q2':
        if dates.curmon >= 4:
            relyear = relyear + 1
    elif qx == 'Q3':
        if dates.curmon >= 7:
            relyear = relyear + 1
    elif qx == 'Q4':
        if dates.curyear >= 10:
            relyear = relyear + 1

    return relyear


def cal_contracts(c):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of cal values (eg Cal15)
    """
    years = list(set([x.year for x in c.columns]))

    dfs = []
    for year in years:
        s = c[[x for x in c.columns if x.year == year]].dropna(how='all', axis=1)
        if len(s.columns) == 12: # only do if we have full set of contracts
            s = s.mean(axis=1)
            s.name = 'CAL {}'.format(year)
            dfs.append(s)

    res = pd.concat(dfs, 1)
    # sort columns by years
    cols = list(res.columns)
    cols.sort(key=lambda s: s.split()[1])
    res = res[cols]
    return res


def cal_spreads(q):
    """
    Given a dataframe of cal contract values (eg CAL 2015, CAL 2020)
    with columns headings as 'CAL 2015', 'CAL 2020'
    Return a dataframe of cal spreads (eg CAL 2015-2016)
    Does Q1-Q2, Q2-Q3, Q3-Q4, Q4-Q1
    """

    calspr = []
    for col in q.columns:
        colcal = col.split(' ')[0]
        colcalyr = col.split(' ')[1]

        curyear = int(colcalyr)
        nextyear = curyear + 1

        colcalnextyr = 'CAL %s' % (nextyear)
        if colcalnextyr in q.columns:
            r = q[col] - q[colcalnextyr]
            r.name = 'CAL {}-{}'.format(curyear, nextyear)
            calspr.append(r)

    res = pd.concat(calspr, 1, sort=True)
    return res