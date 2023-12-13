"""
Utility for forward contracts
"""
import re
from calendar import month_abbr

import numpy as np
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
    12: "Z",
}

futures_month_conv_inv = {v: k for k, v in futures_month_conv.items()}


def convert_contract_to_date(contract):
    """
    Given a string like FB_2020J return 2020-01-01
    :param contract:
    :return:
    """
    c = re.findall("\d\d\d\d\w", contract)
    if len(c) > 0:
        c = c[0]
    d = "%s-%s-1" % (c[:4], futures_month_conv_inv.get(c[-1], 0))
    return d


def convert_columns_to_date(contracts: pd.DataFrame) -> pd.DataFrame:
    remap = {}
    for col in contracts.columns:
        try:
            remap[col] = pd.to_datetime(convert_contract_to_date(col))
        except IndexError as _:
            pass
        except TypeError as _:
            pass
    contracts = contracts.rename(columns=remap)
    return contracts


def time_spreads_monthly(contracts, m1, m2):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of time spreads  (eg m1 = 12, m2 = 12 gives Dec-Dec spread)
    """

    contracts = convert_columns_to_date(contracts)

    cf = [x for x in contracts if x.month == m1]
    dfs = []

    for c1 in cf:
        year1, year2 = c1.year, c1.year
        if m2 <= m1:
            year2 = year1 + 1
        c2 = [x for x in contracts if x.month == m2 and x.year == year2]
        if len(c2) == 1:
            c2 = c2[0]
            s = contracts[c1] - contracts[c2]
            s.name = year1
            dfs.append(s)

    res = pd.concat(dfs, axis=1)
    res = res.dropna(how="all", axis="rows")
    return res


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


def fly(contracts, m1, m2, m3):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of flys  (eg m1 = 1, m2 = 2, m3 = 3 gives Jan/Feb/Mar fly)
    """
    contracts = convert_columns_to_date(contracts)

    cf = [x for x in contracts if x.month == m1]
    dfs = []
    for c1 in cf:
        year1, year2, year3 = c1.year, c1.year, c1.year
        # year rollover
        if m2 < m1:  # eg dec/jan/feb, make jan y+1
            year2 = year2 + 1
        if m3 < m1:
            year3 = year3 + 1
        c2 = [x for x in contracts if x.month == m2 and x.year == year2]
        c3 = [x for x in contracts if x.month == m3 and x.year == year3]
        if len(c2) == 1 and len(c3) == 1:
            c2, c3 = c2[0], c3[0]
            s = contracts[c1] + contracts[c3] - (2 * contracts[c2])
            s.name = year1
            dfs.append(s)

    res = pd.concat(dfs, axis=1)
    res = res.dropna(how="all", axis="rows")
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


def time_spreads(contracts, m1, m2):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of time spreads  (eg m1 = 12, m2 = 12 gives Dec-Dec spread)
    """
    if isinstance(m1, int) and isinstance(m2, int):
        return time_spreads_monthly(contracts, m1, m2)

    if m1.lower().startswith("q") and m2.lower().startswith("q"):
        return time_spreads_quarterly(contracts, m1, m2)


def half_year_contracts(contracts):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of half year values (eg H115)
    :param contracts:
    :return:
    """
    contracts = convert_columns_to_date(contracts)
    years = list(set([x.year for x in contracts.columns]))

    dfs = []
    for year in years:
        c1, c2, c3, c4, c5, c6 = (
            "{}-01-01".format(year),
            "{}-02-01".format(year),
            "{}-03-01".format(year),
            "{}-04-01".format(year),
            "{}-05-01".format(year),
            "{}-06-01".format(year),
        )
        if (
            c1 in contracts.columns
            and c2 in contracts.columns
            and c3 in contracts.columns
            and c4 in contracts.columns
            and c5 in contracts.columns
            and c6 in contracts.columns
        ):
            s = (
                pd.concat(
                    [
                        contracts[c1],
                        contracts[c2],
                        contracts[c3],
                        contracts[c4],
                        contracts[c5],
                        contracts[c6],
                    ],
                    axis=1,
                )
                .dropna(how="any")
                .mean(axis=1)
            )
            s.name = "H1 {}".format(year)
            dfs.append(s)
        c7, c8, c9, c10, c11, c12 = (
            "{}-07-01".format(year),
            "{}-08-01".format(year),
            "{}-09-01".format(year),
            "{}-10-01".format(year),
            "{}-11-01".format(year),
            "{}-12-01".format(year),
        )
        if (
            c7 in contracts.columns
            and c8 in contracts.columns
            and c9 in contracts.columns
            and c10 in contracts.columns
            and c11 in contracts.columns
            and c12 in contracts.columns
        ):
            s = (
                pd.concat(
                    [
                        contracts[c7],
                        contracts[c8],
                        contracts[c9],
                        contracts[c10],
                        contracts[c11],
                        contracts[c12],
                    ],
                    axis=1,
                )
                .dropna(how="any")
                .mean(axis=1)
            )
            s.name = "H2 {}".format(year)
            dfs.append(s)
        # Winter
        c1, c2, c3, c10, c11, c12 = (
            "{}-01-01".format(year + 1),
            "{}-02-01".format(year + 1),
            "{}-03-01".format(year + 1),
            "{}-10-01".format(year),
            "{}-11-01".format(year),
            "{}-12-01".format(year),
        )
        if (
            c1 in contracts.columns
            and c2 in contracts.columns
            and c3 in contracts.columns
            and c10 in contracts.columns
            and c11 in contracts.columns
            and c12 in contracts.columns
        ):
            s = (
                pd.concat(
                    [
                        contracts[c1],
                        contracts[c2],
                        contracts[c3],
                        contracts[c10],
                        contracts[c11],
                        contracts[c12],
                    ],
                    axis=1,
                )
                .dropna(how="any")
                .mean(axis=1)
            )
            s.name = "Winter {}".format(year)
            dfs.append(s)
            # Summer
            c4, c5, c6, c7, c8, c9 = (
                "{}-04-01".format(year),
                "{}-05-01".format(year),
                "{}-06-01".format(year),
                "{}-07-01".format(year),
                "{}-08-01".format(year),
                "{}-09-01".format(year),
            )
            if (
                c4 in contracts.columns
                and c5 in contracts.columns
                and c6 in contracts.columns
                and c7 in contracts.columns
                and c8 in contracts.columns
                and c9 in contracts.columns
            ):
                s = (
                    pd.concat(
                        [
                            contracts[c4],
                            contracts[c5],
                            contracts[c6],
                            contracts[c7],
                            contracts[c8],
                            contracts[c9],
                        ],
                        axis=1,
                    )
                    .dropna(how="any")
                    .mean(axis=1)
                )
                s.name = "Summer {}".format(year)
                dfs.append(s)

    res = pd.concat(dfs, axis=1)
    # sort columns by years
    cols = list(res.columns)
    cols.sort(key=lambda s: s.split()[1])
    res = res[cols]
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


def half_year_spreads(q):
    """
    Given a dataframe of half year values (eg Brent H115, Brent H215, Brent H116)
    with columns headings as 'H1 2015', 'H2 2015'
    Return a dataframe of half year spreads (eg H1-H2 15, H2-H1 15)

    """

    half_year_spread = []
    for col in q.columns:
        colhx = col.split(" ")[0]
        colhxyr = col.split(" ")[1]
        if colhx == "H2":
            colhxyr = int(colhxyr) + 1
        colqy = f"H2 {colhxyr}" if colhx == "H1" else f"H1 {colhxyr}"
        if colqy in q.columns:
            r = q[col] - q[colqy]
            r.name = "{}{} {}".format(colhx, colqy.split(" ")[0], col.split(" ")[1])
            half_year_spread.append(r)

    for col in q.columns:
        colhx = col.split(" ")[0]
        colhxyr = col.split(" ")[1]
        if colhx == "Summer":
            colqy = f"Winter {colhxyr}"
            if colqy in q.columns:
                r = q[col] - q[colqy]
                r.name = "{}{} {}".format(colhx, colqy.split(" ")[0], col.split(" ")[1])
                half_year_spread.append(r)
        if colhx == "Winter":
            colqy = f"Summer {int(colhxyr) + 1}"
            if colqy in q.columns:
                r = q[col] - q[colqy]
                r.name = "{}{} {}".format(colhx, colqy.split(" ")[0], col.split(" ")[1])
                half_year_spread.append(r)

    res = pd.concat(half_year_spread, axis=1, sort=True)
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


def cal_contracts(contracts):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of cal values (eg Cal15)
    """

    contracts = convert_columns_to_date(contracts)
    years = list(set([x.year for x in contracts.columns]))

    dfs = []
    for year in years:
        s = contracts[[x for x in contracts.columns if x.year == year]].dropna(
            how="all", axis=1
        )
        if len(s.columns) == 12:  # only do if we have full set of contracts
            s = s.mean(axis=1)
            s.name = "CAL {}".format(year)
            dfs.append(s)
        elif (
            year == dates.curyear and len(s.columns) > 0
        ):  # sometimes current year passed in has less than 12 columns but should be included
            s = s.mean(axis=1)
            s.name = "CAL {}".format(year)
            dfs.append(s)

    res = pd.concat(dfs, axis=1)
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
        colcalyr = col.split(" ")[1]

        curyear = int(colcalyr)
        nextyear = curyear + 1

        colcalnextyr = "CAL %s" % (nextyear)
        if colcalnextyr in q.columns:
            r = q[col] - q[colcalnextyr]
            r.name = "CAL {}-{}".format(curyear, nextyear)
            calspr.append(r)

    if len(calspr) > 0:
        res = pd.concat(calspr, axis=1, sort=True)
        return res


def spread_combinations(contracts):
    output = {}
    output["Calendar"] = cal_contracts(contracts)
    output["Calendar Spread"] = cal_spreads(output["Calendar"])
    output["Quarterly"] = quarterly_contracts(contracts)
    output["Half Year"] = half_year_contracts(contracts)

    q = output["Quarterly"]
    for qx in ["Q1", "Q2", "Q3", "Q4"]:
        output[qx] = q[[x for x in q if qx in x]]
    output["Quarterly Spread"] = quarterly_spreads(q)
    q = output["Quarterly Spread"]
    for qx in ["Q1Q2", "Q2Q3", "Q3Q4", "Q4Q1"]:
        output[qx] = q[[x for x in q if qx in x]]

    output["Half Year Spread"] = half_year_spreads(output["Half Year"])

    contracts = convert_columns_to_date(contracts)
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        output[month] = contracts[[x for x in contracts.columns if x.month == month]]

    for spread in [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 1],
        [6, 6],
        [6, 12],
        [12, 12],
        [10, 12],
        [4, 9],
        [10, 3],
    ]:
        tag = "%s%s" % (month_abbr[spread[0]], month_abbr[spread[1]])
        output[tag] = time_spreads(contracts, spread[0], spread[1])

    for flyx in [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11, 12],
        [11, 12, 1],
        [12, 1, 2],
    ]:
        tag = "%s%s%s" % (month_abbr[flyx[0]], month_abbr[flyx[1]], month_abbr[flyx[2]])
        output[tag] = fly(contracts, flyx[0], flyx[1], flyx[2])

    return output


def spread_combination(contracts, combination_type, verbose_columns=True):
    """
    Convenience method to access functionality in forwards using a combination_type keyword
    :param contracts:
    :param combination_type:
    :return:
    """
    combination_type = combination_type.lower()
    contracts = contracts.dropna(how="all", axis="rows")

    if combination_type == "calendar":
        c_contracts = cal_contracts(contracts)
        colmap = dates.find_year(c_contracts)
        c_contracts = c_contracts.rename(
            columns={x: colmap[x] for x in c_contracts.columns}
        )
        return c_contracts
    if combination_type == "calendar spread":
        c_contracts = cal_spreads(cal_contracts(contracts))
        if not verbose_columns:
            colmap = dates.find_year(c_contracts)
            c_contracts = c_contracts.rename(
                columns={x: colmap[x] for x in c_contracts.columns}
            )
        return c_contracts
    if combination_type == "half year":
        c_contracts = half_year_contracts(contracts)
        return c_contracts
    if combination_type == "half year spread":
        c_contracts = half_year_spreads(half_year_contracts(contracts))
        return c_contracts

    if combination_type.startswith("q"):
        q_contracts = quarterly_contracts(contracts)
        m = re.search("q\dq\dq\d", combination_type)
        if m:
            q_spreads = fly_quarterly(
                q_contracts,
                x=int(combination_type[1]),
                y=int(combination_type[3]),
                z=int(combination_type[5]),
            )
            if not verbose_columns:
                colmap = dates.find_year(q_spreads)
                q_spreads = q_spreads.rename(
                    columns={x: colmap[x] for x in q_spreads.columns}
                )
            return q_spreads
        m = re.search("q\dq\d", combination_type)
        if m:
            q_spreads = time_spreads_quarterly(
                contracts, combination_type[0:2], combination_type[2:4]
            )
            if verbose_columns:
                colmap = dates.find_year(q_spreads)
                q_spreads = q_spreads.rename(
                    columns={
                        x: "%s %s" % (combination_type.upper(), colmap[x])
                        for x in q_spreads.columns
                    }
                )
            return q_spreads

        m = re.search("q\d", combination_type)
        if m:
            q_contracts = q_contracts[
                [
                    x
                    for x in q_contracts.columns
                    if x.startswith(combination_type.upper())
                ]
            ]
            if not verbose_columns:
                colmap = dates.find_year(q_contracts)
                q_contracts = q_contracts.rename(
                    columns={x: colmap[x] for x in q_contracts.columns}
                )
            return q_contracts

    # handle monthly, spread and fly inputs
    contracts = convert_columns_to_date(contracts)
    month_abbr_inv = {
        month.lower(): index for index, month in enumerate(month_abbr) if month
    }
    months = [x.lower() for x in month_abbr]
    if len(combination_type) == 3 and combination_type in months:
        c = contracts[
            [x for x in contracts if x.month == month_abbr_inv[combination_type]]
        ]
        if verbose_columns:
            c = c.rename(columns={x: x.strftime("%b %Y") for x in c.columns})
        else:
            c = c.rename(columns={x: x.year for x in c.columns})
        return c
    if len(combination_type) == 6:
        m1, m2 = combination_type[0:3], combination_type[3:6]
        if m1 in months and m2 in months:
            c = time_spreads(contracts, month_abbr_inv[m1], month_abbr_inv[m2])
            if verbose_columns:
                c = c.rename(
                    columns={
                        x: "%s%s %s" % (m1.title(), m2.title(), x) for x in c.columns
                    }
                )
            return c
    if len(combination_type) == 9:
        m1, m2, m3 = combination_type[0:3], combination_type[3:6], combination_type[6:9]
        if m1 in months and m2 in months and m3 in months:
            c = fly(
                contracts, month_abbr_inv[m1], month_abbr_inv[m2], month_abbr_inv[m3]
            )
            if verbose_columns:
                c = c.rename(
                    columns={
                        x: "%s%s%s %s" % (m1.title(), m2.title(), m3.title(), x)
                        for x in c.columns
                    }
                )
            return c


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def extract_expiry_date(contract, expiry_dates):
    return expiry_dates.get(contract, contract + pd.offsets.MonthEnd(1))


def determine_roll_date(df, expiry_date, roll_days):
    cdf = df.copy().dropna(how="all", axis="rows") # remove non-trading days
    if expiry_date in cdf.index:
        idx_position = cdf.index.get_loc(expiry_date)
        new_idx_position = idx_position - roll_days

        if new_idx_position >= 0:
            return cdf.index[new_idx_position]

    return expiry_date


def continuous_futures(df, expiry_dates=None, roll_days=0, front_month=1, back_adjust=False) -> pd.DataFrame:
    """
    Create a continuous future from individual contracts by stitching together contracts after they expire
    with an option for back-adjustment.

    :param df: DataFrame with individual contracts as columns.
    :param expiry_dates: Dictionary mapping contract dates to their respective expiry dates.
    :param roll_days: Number of days before the expiry date to roll to the next contract.
    :param front_month: Determines which contract month to select.
    :param back_adjust: If True, apply back-adjustment to the prices.
    :return: DataFrame representing the continuous future.
    """
    mask_switch = pd.DataFrame(index=df.index, columns=df.columns)
    mask_adjust = pd.DataFrame(index=df.index, columns=df.columns)

    df.columns = [pd.to_datetime(x) for x in df.columns]

    # Format expiry_dates if provided
    if expiry_dates:
        expiry_dates = {
            pd.to_datetime(x): pd.to_datetime(expiry_dates[x]) for x in expiry_dates
        }

    # Iterating over the columns (contracts)
    for contract in df.columns:
        prev_contract = contract - pd.offsets.MonthBegin(1)
        next_contract = contract + pd.offsets.MonthBegin(1)

        # Determine expiry date for each contract
        expiry_date = extract_expiry_date(contract, expiry_dates)
        prev_contract_expiry_date = extract_expiry_date(prev_contract, expiry_dates)

        # Adjust expiry date based on roll_days
        roll_date = determine_roll_date(df, expiry_date, roll_days)
        prev_contract_roll_date = determine_roll_date(df, prev_contract_expiry_date, roll_days)

        # Set the cells to 1 where the index date is between the current contract date and the adjusted expiry date
        mask_switch.loc[
            (mask_switch.index > pd.Timestamp(prev_contract_roll_date))
            & (mask_switch.index <= pd.Timestamp(roll_date)),
            contract,
        ] = 1

        # Keep a track of difference between front and back contract on roll date
        if roll_date in df.index and  contract in df.columns and next_contract in df.columns:
            adj_value = df.at[roll_date, next_contract] - df.at[roll_date, contract]
            mask_adjust.loc[
                (mask_switch.index > prev_contract_roll_date)
                & (mask_switch.index <= roll_date),
                contract,
            ] = adj_value

    mask_switch = mask_switch.shift(front_month - 1, axis=1)  # handle front month eg M2, M3 etc
    # Multiply df with mask and sum along the rows
    continuous_df = df.mul(mask_switch, axis=1).sum(axis=1, skipna=True, min_count=1)
    continuous_df = pd.DataFrame(continuous_df, columns=[f"M{front_month}"])

    # Back-adjustment
    if back_adjust:
        mask_adjust_series = mask_adjust.fillna(method='bfill').sum(axis=1, skipna=True, min_count=1).fillna(0)
        continuous_df = continuous_df.add(mask_adjust_series, axis=0)

    # Store mask in attributes for reference
    continuous_df.attrs["mask_switch"] = mask_switch
    continuous_df.attrs["mask_adjust"] = mask_adjust

    return continuous_df


# if __name__ == "__main__":
      # from pylim import lim
#
#     df = lim.series(["CL_2023Z", "CL_2024F"])
#     spread_combination(df, "DecJan")
