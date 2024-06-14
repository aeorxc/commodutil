"""
Utility for forward contracts
"""
import re
from calendar import month_abbr, monthrange
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List

from commodutil.forward.calendar import cal_contracts, cal_spreads, half_year_contracts, half_year_spreads
from commodutil.forward.continuous import generate_series
from commodutil.forward.fly import fly, all_fly_spreads, fly_combos
from commodutil.forward.quarterly import quarterly_contracts, all_quarterly_rolls, time_spreads_quarterly, \
    fly_quarterly, all_quarterly_flys
from commodutil.forward.spreads import time_spreads_monthly, all_monthly_spreads, monthly_spread_combos_extended
from commodutil.forward.util import convert_columns_to_date, month_abbr_inv

from commodutil import dates


def time_spreads(contracts, m1, m2):
    """
    Given a dataframe of daily values for monthly contracts (eg Brent Jan 15, Brent Feb 15, Brent Mar 15)
    with columns headings as '2020-01-01', '2020-02-01'
    Return a dataframe of time spreads  (eg m1 = 12, m2 = 12 gives Dec-Dec spread)
    """
    if isinstance(m1, int) and isinstance(m2, int):
        return time_spreads_monthly(contracts, m1, m2, col_format="%Y")

    if m1.lower().startswith("q") and m2.lower().startswith("q"):
        return time_spreads_quarterly(contracts, m1, m2)


def all_spread_combinations(contracts):
    output = {}
    output["Calendar"] = cal_contracts(contracts)
    output["Calendar Spread"] = cal_spreads(output["Calendar"])
    output["Quarterly"] = quarterly_contracts(contracts)
    output["Half Year"] = half_year_contracts(contracts)

    q = output["Quarterly"]
    for qx in ["Q1", "Q2", "Q3", "Q4"]:
        output[qx] = q[[x for x in q if qx in x]]
    output["Quarterly Spread"] = all_quarterly_rolls(q)
    q = output["Quarterly Spread"]
    for qx in ["Q1Q2", "Q2Q3", "Q3Q4", "Q4Q1"]:
        output[qx] = q[[x for x in q if qx in x]]

    output["Half Year Spread"] = half_year_spreads(output["Half Year"])

    contracts = convert_columns_to_date(contracts)
    for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        output[month] = contracts[[x for x in contracts.columns if x.month == month]]

    for spread in monthly_spread_combos_extended:
        tag = "%s%s" % (month_abbr[spread[0]], month_abbr[spread[1]])
        output[tag] = time_spreads(contracts, spread[0], spread[1])

    for flyx in fly_combos:
        tag = "%s%s%s" % (month_abbr[flyx[0]], month_abbr[flyx[1]], month_abbr[flyx[2]])
        output[tag] = fly(contracts, flyx[0], flyx[1], flyx[2])

    return output


def replace_last_month_with_nan(series):
    # Find the last valid month
    series_dropped_na = series.dropna()
    if series_dropped_na.empty:
        return series
    last_month = pd.to_datetime(f"{series_dropped_na.index[-1].year}-{series_dropped_na.index[-1].month}-01")
    _, last_day = monthrange(last_month.year, last_month.month)
    last_valid_month_end = pd.to_datetime(f"{last_month.year}-{last_month.month}-{last_day}")
    # Replace series with NaN for the last valid month
    if last_month.year < dates.curyear:
        series[last_month:last_valid_month_end] = np.nan

    return series


def spread_combination_quarter(contracts, combination_type, verbose_columns=True, exclude_price_month=False,
                               col_format=None):
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
            if exclude_price_month:
                contracts = q_spreads.apply(replace_last_month_with_nan, axis=0)
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
            if exclude_price_month:
                contracts = q_spreads.apply(replace_last_month_with_nan, axis=0)
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


def spread_combination_fly(contracts, combination_type, verbose_columns=True, exclude_price_month=False,
                           col_format=None):
    months = [x.lower() for x in month_abbr]
    if exclude_price_month:
        contracts = contracts.apply(replace_last_month_with_nan, axis=0)
    m1, m2, m3 = combination_type[0:3], combination_type[3:6], combination_type[6:9]
    if m1 in months and m2 in months and m3 in months:
        c = fly(
            contracts, month_abbr_inv[m1], month_abbr_inv[m2], month_abbr_inv[m3]
        )
        # if verbose_columns:
        #     c = c.rename(
        #         columns={
        #             x: "%s%s%s %s" % (m1.title(), m2.title(), m3.title(), x)
        #             for x in c.columns
        #         }
        #     )
        if exclude_price_month:
            contracts = c.apply(replace_last_month_with_nan, axis=0)
        return c


def spread_combination_month(contracts, combination_type, verbose_columns=True, exclude_price_month=False,
                             col_format=None):
    months = [x.lower() for x in month_abbr]
    m1, m2 = combination_type[0:3], combination_type[3:6]
    if m1 in months and m2 in months:
        c = time_spreads(contracts, month_abbr_inv[m1], month_abbr_inv[m2])
        if verbose_columns:
            c = c.rename(
                columns={
                    x: "%s%s %s" % (m1.title(), m2.title(), x) for x in c.columns
                }
            )
        if exclude_price_month:
            contracts = c.apply(replace_last_month_with_nan, axis=0)
        return c


def spread_combination(contracts, combination_type, verbose_columns=True, exclude_price_month=False, col_format=None):
    """
    Convenience method to access functionality in forwards using a combination_type keyword
    :param contracts:
    :param combination_type:
    :return:
    """
    combination_type = combination_type.lower().replace(" ", "")
    contracts = contracts.dropna(how="all", axis="rows")

    if combination_type == "calendar":
        c_contracts = cal_contracts(contracts, col_format="%Y")
        return c_contracts
    if combination_type == "calendarspread":
        c_contracts = cal_spreads(cal_contracts(contracts), col_format=col_format)
        return c_contracts
    if combination_type == "halfyear":
        c_contracts = half_year_contracts(contracts)
        return c_contracts
    if combination_type == "halfyearspread":
        c_contracts = half_year_spreads(half_year_contracts(contracts))
        return c_contracts

    if combination_type.startswith("monthly"):
        if col_format is None:
            col_format = "%b%b %y"
        return all_monthly_spreads(contracts, col_format=col_format)

    if combination_type.startswith("fly"):
        if col_format is None:
            col_format = "%b%b%b %y"
        return all_fly_spreads(contracts, col_format=col_format)

    if combination_type.startswith("quarterlyroll"):
        if col_format is None:
            col_format = "%q%q %y"
        return all_quarterly_rolls(quarterly_contracts(contracts), col_format=col_format)

    if combination_type.startswith("quarterlyfly"):
        if col_format is None:
            col_format = "%q%q%q %y"
        return all_quarterly_flys(quarterly_contracts(contracts), col_format=col_format)

    if combination_type.startswith("quarterly"):
        if col_format is None:
            col_format = "%q %y"
        return quarterly_contracts(contracts, col_format=col_format)

    if combination_type.startswith("q"):
        return spread_combination_quarter(contracts, combination_type=combination_type, verbose_columns=verbose_columns,
                                          exclude_price_month=exclude_price_month, col_format=col_format)

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
    if len(combination_type) == 6:  # spread
        return spread_combination_month(contracts, combination_type, verbose_columns=verbose_columns,
                                        exclude_price_month=exclude_price_month, col_format=col_format)
    if len(combination_type) == 9:  # fly
        return spread_combination_fly(contracts, combination_type, verbose_columns=verbose_columns,
                                      exclude_price_month=exclude_price_month, col_format=col_format)


def filter_columns_by_date(contract_data, start_date, end_date):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    # return [col for col in contract_data.columns if start_date <= col.to_pydatetime().date() <= end_date]
    return [col for col in contract_data.columns
            if start_date <= (col.date() if isinstance(col, pd.Timestamp) else col) <= end_date]


def recent_structure(contracts: pd.DataFrame, structure_combo: List[List[int]] = [[1, 2], [1, 3], [1, 6], [1, 12]],
                     roll_days: int = 0) -> pd.DataFrame:
    """
    Given a list of contracts, calculate structure.

    :param contracts: DataFrame with individual contracts as columns.
    :param structure_combo: List of lists where each sublist represents a structure to calculate.
    :param roll_days: Number of days before the expiry date to roll to the next contract.
    :return: DataFrame representing the calculated structure for each structure combo.
    """
    contracts = convert_columns_to_date(contracts)

    # Flatten and unique structure_combo using list comprehension
    structure_months = list(set(month for sublist in structure_combo for month in sublist))

    dfs = []
    for st in structure_months:
        s = generate_series(contracts, front_month=st, roll_days=roll_days)
        s = s.rename(columns={x: f"M{st}" for x in s.columns})
        s.attrs = {}
        dfs.append(s)

    df = pd.concat(dfs, axis=1)

    # Calculate M1-M2, M1-M3, M1-M6
    for st in structure_combo:
        df[f"M{st[0]}-M{st[1]}"] = df[f"M{st[0]}"].sub(df[f"M{st[1]}"]).dropna()

    drop_columns = {x for x in df.columns if x not in [f"M{st[0]}-M{st[1]}" for st in structure_combo]}
    df = df.drop(columns=drop_columns)

    df = df.dropna(how="all", axis="rows")

    return df


def recent_spreads(contracts: pd.DataFrame, combination_type: str, **kwargs):
    """Given a list of contracts, filter the contracts to the list of most recent relevant contracts"""
    if contracts is None:
        return None

    if combination_type == 'structure':
        return recent_structure(contracts, **kwargs)

    contracts = convert_columns_to_date(contracts)
    current_date = datetime.now().date()  # Convert to datetime.date
    start_date = current_date - relativedelta(months=3)  # Approximating a month as 30 days
    end_date = current_date + relativedelta(months=6)  # Approximating a month as 30 days
    filtered_columns = filter_columns_by_date(contracts, start_date, end_date)

    start_date_qtr = current_date - relativedelta(months=5)
    end_date_qtr = current_date + relativedelta(months=13)
    end_date_qtr2 = current_date + relativedelta(months=16)
    filtered_columns_qtr = filter_columns_by_date(contracts, start_date_qtr, end_date_qtr)
    filtered_columns_qtr2 = filter_columns_by_date(contracts, start_date_qtr, end_date_qtr2)

    if combination_type == "contracts":
        month_df = contracts[filtered_columns]
        month_df.columns = [f"{col.strftime('%b%y').lower()}" for col in month_df.columns]
        return month_df
    elif combination_type in ['monthly']:
        spread_df = spread_combination(contracts[filtered_columns], combination_type="monthly",
                                       col_format="%b%b %y")

        return spread_df

    elif combination_type in ['quarterly']:
        spread_df = spread_combination(contracts=contracts[filtered_columns_qtr],
                                       combination_type="quarterly",
                                       col_format="%q %y")

        return spread_df

    elif combination_type in ['quarterly roll']:
        spread_df = spread_combination(contracts=contracts[filtered_columns_qtr],
                                       combination_type="quarterly roll",
                                       col_format="%q%q %y")

        return spread_df
    elif combination_type in ['quarterly fly']:
        spread_df = spread_combination(contracts=contracts[filtered_columns_qtr2],
                                       combination_type="quarterly fly",
                                       col_format="%q%q%q %y")

        return spread_df

    elif combination_type in ['fly']:
        spread_df = spread_combination(contracts=contracts[filtered_columns], combination_type="fly",
                                       col_format="%b%b%b %y")
        return spread_df


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# if __name__ == "__main__":
# from pylim import lim
#
#     df = lim.series(["CL_2023Z", "CL_2024F"])
#     spread_combination(df, "DecJan")
