import datetime
from calendar import month_abbr

import pandas as pd
from commodutil.forward.util import convert_columns_to_date

fly_combos = [
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
]


def fly(contracts, m1, m2, m3, col_format=None):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of flys  (eg m1 = 1, m2 = 2, m3 = 3 gives Jan/Feb/Mar fly)
    """
    contracts = convert_columns_to_date(contracts)

    cf = [x for x in contracts if x.month == m1]
    dfs = []
    legmap = {}
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
            if col_format is not None:
                if col_format == "%Y":
                    s.name = year1
            else:
                s.name = f"{month_abbr[m1]}{month_abbr[m2]}{month_abbr[m3]} {year1}"
            legmap[s.name] = [c1, c2, c3]
            dfs.append(s)

    res = pd.concat(dfs, axis=1)
    res = res.dropna(how="all", axis="rows")
    res.attrs = legmap
    return res


def all_fly_spreads(contracts, start_date=None, end_date=None, col_format=None):
    dfs = []
    for flyx in fly_combos:
        df = fly(contracts, flyx[0], flyx[1], flyx[2])
        dfs.append(df)

    res = pd.concat(dfs, axis=1)

    legmap = {}  # TODO move this to a function reduplicates
    for df in dfs:
        legmap.update(df.attrs)
    res.attrs['legmap'] = legmap

    def parse_date(col_name):
        month_str, year = col_name.split()  # Splitting by space to separate month(s) and year
        month_abbr = month_str[:3]  # Taking the first three letters as the month abbreviation
        month = datetime.datetime.strptime(month_abbr, "%b").month  # Convert abbreviation to month number
        return datetime.datetime(year=int(year), month=month, day=1)

    columns = res.columns
    if start_date is not None:
        columns = [col for col in columns if parse_date(col).date() >= start_date]
    if end_date is not None:
        columns = [col for col in columns if parse_date(col).date() <= end_date]

    res = res[columns]

    if col_format is not None and 'legmap' in res.attrs:  # TODO move this to a function reduplicates
        if col_format == "%b%y":
            col_mapping = {
                col: f"{leg[0].strftime(col_format)}{leg[1].strftime(col_format)}{leg[2].strftime(col_format)}" for
                col, leg in
                legmap.items()}
        elif col_format == "%b%b%b %y":
            col_mapping = {
                col: f"{leg[0].strftime('%b')}{leg[1].strftime('%b')}{leg[2].strftime('%b')} {leg[0].strftime('%y')}"
                for col, leg in legmap.items()}
        res = res.rename(columns=col_mapping)

    return res
