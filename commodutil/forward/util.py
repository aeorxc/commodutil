import datetime
import re
from calendar import month_abbr
import pandas as pd

month_abbr_inv = {
    month.lower(): index for index, month in enumerate(month_abbr) if month
}


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
            if isinstance(col, datetime.date):
                remap[col] = pd.to_datetime(col)
            else:
                remap[col] = pd.to_datetime(convert_contract_to_date(col))
        except IndexError as _:
            pass
        except TypeError as _:
            pass
    contracts = contracts.rename(columns=remap)
    return contracts


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


def extract_expiry_dates_from_contracts(contracts):
    "Given a dataframe of contracts use the value of the last date in each given contract as the expiry date"
    "This is used when we don't have an explicit map of contracts to expiry dates"
    expiry_dates = {}
    unique_expiry_dates = {}
    for contract in contracts.columns:
        # Find the last non-null date for the contract
        last_date = contracts[contract].dropna()
        if len(last_date) > 0:
            last_date = last_date.index[-1]
            expiry_dates[contract] = last_date
            # Check if the expiry date is already in the unique_expiry_dates values
            if last_date not in unique_expiry_dates.values():
                unique_expiry_dates[contract] = last_date

    return unique_expiry_dates
