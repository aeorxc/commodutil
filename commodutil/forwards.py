"""
Utility for forward contracts
"""
import re
import pandas as pd
from calendar import month_abbr
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
    d = '%s-%s-1' % (c[:4], futures_month_conv_inv.get(c[-1], 0))
    return d


def time_spreads_monthly(contracts, m1, m2):
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
    res = res.dropna(how='all', axis='rows')
    return res


def time_spreads_quarterly(contracts, m1, m2):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of time spreads  (eg m1 = Q1, m2 = Q2 gives Q1-Q2 spread)
    """

    qtrcontracts = quarterly_contracts(contracts)
    qtrcontracts_years = dates.find_year(qtrcontracts)
    cf = [x for x in qtrcontracts if x.startswith(m1)]
    dfs = []

    for c1 in cf:
        year1, year2 = qtrcontracts_years[c1], qtrcontracts_years[c1]
        if int(m1[-1]) >= int(m2[-1]): # eg Q1-Q1 or Q4-Q1, then do Q419 - Q120 (year ahead)
            year2 = year1 + 1
        c2 = [x for x in qtrcontracts if x.startswith(m2) and qtrcontracts_years[x] == year2]
        if len(c2) == 1:
            c2 = c2[0]
            s = qtrcontracts[c1] - qtrcontracts[c2]
            s.name = year1
            dfs.append(s)

    res = pd.concat(dfs, 1)
    res = res.dropna(how='all', axis='rows')
    return res


def fly(contracts, m1, m2, m3):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of flys  (eg m1 = 1, m2 = 2, m3 = 3 gives Jan/Feb/Mar fly)
    """

    cf = [x for x in contracts if x.month == m1]
    dfs = []
    for c1 in cf:
        year1, year2, year3 = c1.year, c1.year, c1.year
        # year rollover
        if m2 < m1: # eg dec/jan/feb, make jan y+1
            year2 = year2 + 1
        if m3 < m1:
            year3 = year3 + 1
        c2 = [x for x in contracts if x.month == m2 and x.year == year2]
        c3 = [x for x in contracts if x.month == m3 and x.year == year3]
        if len(c2) == 1 and len(c3) == 1:
            c2, c3 = c2[0], c3[0]
            s = contracts[c1] + contracts[c3] - (2*contracts[c2])
            s.name = year1
            dfs.append(s)

    res = pd.concat(dfs, 1)
    res = res.dropna(how='all', axis='rows')
    return res


def time_spreads(contracts, m1, m2):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of time spreads  (eg m1 = 12, m2 = 12 gives Dec-Dec spread)
    """
    if isinstance(m1, int) and isinstance(m2, int):
        return time_spreads_monthly(contracts, m1, m2)

    if m1.lower().startswith('q') and m2.lower().startswith('q'):
        return time_spreads_quarterly(contracts, m1, m2)


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
        elif year == dates.curyear and len(s.columns) > 0: # sometimes current year passed in has less than 12 columns but should be included
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
    """

    calspr = []
    for col in q.columns:
        # colcal = col.split(' ')[0]
        colcalyr = col.split(' ')[1]

        curyear = int(colcalyr)
        nextyear = curyear + 1

        colcalnextyr = 'CAL %s' % (nextyear)
        if colcalnextyr in q.columns:
            r = q[col] - q[colcalnextyr]
            r.name = 'CAL {}-{}'.format(curyear, nextyear)
            calspr.append(r)

    if len(calspr) > 0:
        res = pd.concat(calspr, 1, sort=True)
        return res


def spread_combinations(contracts):
    output = {}
    output['Calendar'] = cal_contracts(contracts)
    output['Calendar Spread'] = cal_spreads(output['Calendar'])
    output['Quarterly'] = quarterly_contracts(contracts)

    q = output['Quarterly']
    for qx in ['Q1', 'Q2', 'Q3', 'Q4']:
        output[qx] = q[[x for x in q if qx in x]]
    output['Quarterly Spread'] = quarterly_spreads(q)
    q = output['Quarterly Spread']
    for qx in ['Q1-Q2', 'Q2-Q3', 'Q3-Q4', 'Q4-Q1']:
        output[qx] = q[[x for x in q if qx in x]]

    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        output[month] = contracts[[x for x in contracts.columns if x.month == month]]

    for spread in [[1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10,11], [11,12], [12,1], [6,6], [6,12], [12,12], [10,12], [4,9], [10,3]]:
        tag = '%s%s' % (month_abbr[spread[0]], month_abbr[spread[1]])
        output[tag] = time_spreads(contracts, spread[0], spread[1])

    for flyx in [[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [11,12,1], [12,1,2]]:
        tag = '%s%s%s' % (month_abbr[flyx[0]], month_abbr[flyx[1]], month_abbr[flyx[2]])
        output[tag] = fly(contracts, flyx[0], flyx[1], flyx[2])

    return output


def spread_combination(contracts, combination_type):
    """
    Convenience method to access functionality in forwards using a combination_type keyword
    :param contracts:
    :param combination_type:
    :return:
    """
    combination_type = combination_type.lower()
    contracts = contracts.dropna(how='all', axis='rows')

    if combination_type == "calendar":
        c_contracts = cal_contracts(contracts)
        colmap = dates.find_year(c_contracts)
        c_contracts = c_contracts.rename(columns={x: colmap[x] for x in c_contracts.columns})
        return c_contracts
    if combination_type == "calendar spread":
        c_contracts = cal_spreads(cal_contracts(contracts))
        colmap = dates.find_year(c_contracts)
        c_contracts = c_contracts.rename(columns={x: colmap[x] for x in c_contracts.columns})
        return c_contracts
    if combination_type.startswith('q'):
        q_contracts = quarterly_contracts(contracts)
        m = re.search('q\d-q\d', combination_type)
        if m:
            q_spreads = quarterly_spreads(q_contracts)
            q_spreads = q_spreads[[x for x in q_spreads.columns if x.startswith(combination_type.upper())]]
            colmap = dates.find_year(q_spreads)
            q_spreads = q_spreads.rename(columns={x:colmap[x] for x in q_spreads.columns})
            return q_spreads
        m = re.search('q\d', combination_type)
        if m:
            q_contracts = q_contracts[[x for x in q_contracts.columns if x.startswith(combination_type.upper())]]
            colmap = dates.find_year(q_contracts)
            q_contracts = q_contracts.rename(columns={x:colmap[x] for x in q_contracts.columns})
            return q_contracts

    # handle monthly, spread and fly inputs
    month_abbr_inv = {month.lower(): index for index, month in enumerate(month_abbr) if month}
    months = [x.lower() for x in month_abbr]
    if len(combination_type) == 3 and combination_type in months:
        c = contracts[[x for x in contracts if x.month == month_abbr_inv[combination_type]]]
        c = c.rename(columns={x:x.year for x in c.columns})
        return c
    if len(combination_type) == 6:
        m1, m2 = combination_type[0:3], combination_type[3:6]
        if m1 in months and m2 in months:
            c = time_spreads(contracts, month_abbr_inv[m1], month_abbr_inv[m2])
            return c
    if len(combination_type) == 9:
        m1, m2, m3 = combination_type[0:3], combination_type[3:6], combination_type[6:9]
        if m1 in months and m2 in months and m3 in months:
            c = fly(contracts, month_abbr_inv[m1], month_abbr_inv[m2], month_abbr_inv[m3])
            return c




