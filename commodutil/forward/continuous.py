import pandas as pd

from commodutil.forward.util import extract_expiry_date, determine_roll_date, extract_expiry_dates_from_contracts


def generate_series(df, expiry_dates=None, roll_days=0, front_month=1, back_adjust=False) -> pd.DataFrame:
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

    df = df.dropna(axis=1, how='all')

    # Format expiry_dates if provided
    if expiry_dates:
        expiry_dates = {
            pd.to_datetime(x): pd.to_datetime(expiry_dates[x]) for x in expiry_dates
        }
    else:
        expiry_dates = extract_expiry_dates_from_contracts(contracts=df)

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

            # in some edge cases where we have guessed expiry dates we need to ensure we dont get a scenario where the previous contract expiry date is after the current contract expiry date
            if prev_contract_expiry_date > expiry_date:
                prev_contract_expiry_date = expiry_date - pd.offsets.MonthBegin(1)

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
