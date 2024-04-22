"""
Utility for forward contracts
"""
import re
from calendar import month_abbr, monthrange
import numpy as np
import pandas as pd

from commodutil.forward.calendar import cal_contracts, cal_spreads, half_year_contracts, half_year_spreads
from commodutil.forward.fly import fly, all_fly_spreads, fly_combos
from commodutil.forward.quarterly import quarterly_contracts, all_quarterly_rolls, time_spreads_quarterly, \
    fly_quarterly, all_quarterly_flys
from commodutil.forward.spreads import time_spreads_monthly, all_monthly_spreads, monthly_spread_combos_extended
from commodutil.forward.util import convert_contract_to_date, convert_columns_to_date, month_abbr_inv, futures_month_conv

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


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def extract_expiry_date(contract, expiry_dates):
    if expiry_dates:
        return expiry_dates.get(contract, contract + pd.offsets.MonthEnd(1))

    return contract + pd.offsets.MonthEnd(1)


def determine_roll_date(df, expiry_date, roll_days):
    cdf = df.copy().dropna(how="all", axis="rows")  # remove non-trading days
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
    :param front_month: Determines which contract month(s) to select. Can be an int or list of ints.
    :param back_adjust: If True, apply back-adjustment to the prices.
    :return: DataFrame representing the continuous future for each front month.
    """
    if isinstance(front_month, int):
        front_month = [front_month]  # convert to list if it's a single integer

    df.columns = [pd.to_datetime(x) for x in df.columns]

    # Format expiry_dates if provided
    if expiry_dates:
        expiry_dates = {
            pd.to_datetime(x): pd.to_datetime(expiry_dates[x]) for x in expiry_dates
        }

    continuous_dfs = []

    for front_month_x in front_month:
        mask_switch = pd.DataFrame(index=df.index, columns=df.columns)
        mask_adjust = pd.DataFrame(index=df.index, columns=df.columns)

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
            if roll_date in df.index and contract in df.columns and next_contract in df.columns:
                adj_value = df.at[roll_date, next_contract] - df.at[roll_date, contract]
                mask_adjust.loc[
                    (mask_switch.index > prev_contract_roll_date)
                    & (mask_switch.index <= roll_date),
                    contract,
                ] = adj_value

        mask_switch = mask_switch.shift(front_month_x - 1, axis=1)  # handle front month eg M2, M3 etc
        # Multiply df with mask and sum along the rows
        continuous_df = df.mul(mask_switch, axis=1).sum(axis=1, skipna=True, min_count=1)
        continuous_df = pd.DataFrame(continuous_df, columns=[f"M{front_month_x}"])

        # Back-adjustment
        if back_adjust:
            mask_adjust_series = mask_adjust.fillna(method='bfill').sum(axis=1, skipna=True, min_count=1).fillna(0)
            continuous_df = continuous_df.add(mask_adjust_series, axis=0)

        continuous_dfs.append(continuous_df)

    # Concatenate all dataframes for each front month
    final_df = pd.concat(continuous_dfs, axis=1).dropna(how="all", axis="rows")

    # Store mask in attributes for reference
    final_df.attrs["mask_switch"] = mask_switch
    final_df.attrs["mask_adjust"] = mask_adjust

    return final_df

# if __name__ == "__main__":
# from pylim import lim
#
#     df = lim.series(["CL_2023Z", "CL_2024F"])
#     spread_combination(df, "DecJan")
