import pandas as pd
from commodutil.forward.util import convert_columns_to_date


def relevant_qtr_contract(qx):
    """
    Given a qtr, eg, Q1, determine the right year to use in seasonal charts.
    For example after Feb 2020, use Q1 2021 as Q1 2020 would have stopped pricing

    :param qx:
    :return:
    """
    relyear = dates.curyear
    if qx == "Q1":
        if dates.curmon >= 1:
            relyear = relyear + 1
    elif qx == "Q2":
        if dates.curmon >= 4:
            relyear = relyear + 1
    elif qx == "Q3":
        if dates.curmon >= 7:
            relyear = relyear + 1
    elif qx == "Q4":
        if dates.curyear >= 10:
            relyear = relyear + 1

    return relyear


def time_spreads_quarterly(contracts, m1, m2):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of time spreads  (eg m1 = Q1, m2 = Q2 gives Q1-Q2 spread)
    """
    contracts = convert_columns_to_date(contracts)

    m1, m2 = m1.upper(), m2.upper()
    qtrcontracts = quarterly_contracts(contracts)
    qtrcontracts_years = dates.find_year(qtrcontracts)
    cf = [x for x in qtrcontracts if x.startswith(m1)]
    dfs = []

    for c1 in cf:
        year1, year2 = qtrcontracts_years[c1], qtrcontracts_years[c1]
        if int(m1[-1]) >= int(
                m2[-1]
        ):  # eg Q1-Q1 or Q4-Q1, then do Q419 - Q120 (year ahead)
            year2 = year1 + 1
        c2 = [
            x
            for x in qtrcontracts
            if x.startswith(m2) and qtrcontracts_years[x] == year2
        ]
        if len(c2) == 1:
            c2 = c2[0]
            s = qtrcontracts[c1] - qtrcontracts[c2]
            s.name = year1
            dfs.append(s)

    res = pd.concat(dfs, axis=1)
    res = res.dropna(how="all", axis="rows")
    return res


def quarterly_contracts(contracts):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of quarterly values (eg Q115)
    """
    contracts = convert_columns_to_date(contracts)
    years = list(set([x.year for x in contracts.columns]))

    dfs = []
    for year in years:
        c1, c2, c3 = (
            "{}-01-01".format(year),
            "{}-02-01".format(year),
            "{}-03-01".format(year),
        )
        if (
                c1 in contracts.columns
                and c2 in contracts.columns
                and c3 in contracts.columns
        ):
            s = (
                pd.concat([contracts[c1], contracts[c2], contracts[c3]], axis=1)
                .dropna(how="any")
                .mean(axis=1)
            )
            s.name = "Q1 {}".format(year)
            dfs.append(s)

        c4, c5, c6 = (
            "{}-04-01".format(year),
            "{}-05-01".format(year),
            "{}-06-01".format(year),
        )
        if (
                c4 in contracts.columns
                and c5 in contracts.columns
                and c6 in contracts.columns
        ):
            s = (
                pd.concat(
                    [contracts[c4], contracts[c5], contracts[c6]], axis=1, sort=True
                )
                .dropna(how="any")
                .mean(axis=1)
            )
            s.name = "Q2 {}".format(year)
            dfs.append(s)

        c7, c8, c9 = (
            "{}-07-01".format(year),
            "{}-08-01".format(year),
            "{}-09-01".format(year),
        )
        if (
                c7 in contracts.columns
                and c8 in contracts.columns
                and c9 in contracts.columns
        ):
            s = (
                pd.concat([contracts[c7], contracts[c8], contracts[c9]], axis=1)
                .dropna(how="any")
                .mean(axis=1)
            )
            s.name = "Q3 {}".format(year)
            dfs.append(s)

        c10, c11, c12 = (
            "{}-10-01".format(year),
            "{}-11-01".format(year),
            "{}-12-01".format(year),
        )
        if (
                c10 in contracts.columns
                and c11 in contracts.columns
                and c12 in contracts.columns
        ):
            s = (
                pd.concat([contracts[c10], contracts[c11], contracts[c12]], axis=1)
                .dropna(how="any")
                .mean(axis=1)
            )
            s.name = "Q4 {}".format(year)
            dfs.append(s)

    res = pd.concat(dfs, axis=1)
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
        "Q1": "Q2 {}",
        "Q2": "Q3 {}",
        "Q3": "Q4 {}",
        "Q4": "Q1 {}",
    }

    qtrspr = []
    for col in q.columns:
        colqx = col.split(" ")[0]
        colqxyr = col.split(" ")[1]
        if colqx == "Q4":
            colqxyr = int(colqxyr) + 1
        colqy = sprmap.get(colqx).format(colqxyr)
        if colqy in q.columns:
            r = q[col] - q[colqy]
            r.name = "{}{} {}".format(colqx, colqy.split(" ")[0], col.split(" ")[1])
            qtrspr.append(r)

    res = pd.concat(qtrspr, axis=1, sort=True)
    return res


def fly_quarterly(contracts, x, y, z):
    """
    Given a dataframe of quarterly contract values (eg Brent Q115, Brent Q215, Brent Q315)
    with columns headings as 'Q1 2015', 'Q2 2015'
    Return a dataframe of flys  (eg x = q1 y = q2 z = q3 gives Q1/Q2/Q3 fly)
    """
    contracts = convert_columns_to_date(contracts)

    dfs = []
    cf = [n for n in contracts if "Q%s" % x in n]
    for c1 in cf:
        year1, year2, year3 = int(c1[-4:]), int(c1[-4:]), int(c1[-4:])
        # year rollover

        if x == 4 and y == 1:  # 412 or 413
            year2 = year2 + 1
            year3 = year3 + 1
        if (x == 2 and y == 3 and z == 1) or (x == 2 and y == 3 and z == 1):
            year3 = year3 + 1
        if x == 3 and y == 4:  # 341 or 342
            year3 = year3 + 1

        c2 = [n for n in contracts if "Q%d" % y in n and str(year2) in n]
        c3 = [n for n in contracts if "Q%d" % z in n and str(year3) in n]
        if len(c2) == 1 and len(c3) == 1:
            c2, c3 = c2[0], c3[0]
            s = contracts[c1] + contracts[c3] - (2 * contracts[c2])
            s.name = "Q%dQ%dQ%d %d" % (x, y, z, year1)
            dfs.append(s)

    res = pd.concat(dfs, axis=1)
    res = res.dropna(how="all", axis="rows")
    return res


def quarterly_flys(q):
    """
    Given a dataframe of quarterly contract values (eg Brent Q115, Brent Q215, Brent Q315)
    with columns headings as 'Q1 2015', 'Q2 2015'
    Return a dataframe of quarterly flys (eg Q1Q2Q3)
    Does Q1Q2Q3, Q2Q3Q4, Q3Q4Q1, Q4Q1Q2
    """
    flycombos = ((1, 2, 3), (2, 3, 4), (3, 4, 1), (4, 1, 2))

    dfs = []
    for flycombo in flycombos:
        s = fly_quarterly(contracts=q, x=flycombo[0], y=flycombo[1], z=flycombo[2])
        dfs.append(s)

    res = pd.concat(dfs, axis=1, sort=True)
    return res
