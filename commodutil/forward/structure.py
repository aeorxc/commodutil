import pandas as pd
from commodutil.forward.continuous import generate_series, generate_multiple_continuous_series


def generate_continuous_series(contracts: pd.DataFrame, front_month: int = 1, roll_days: int = 0) -> pd.DataFrame:
    s = generate_series(contracts, front_month=front_month, roll_days=roll_days)
    s = s.rename(columns={x: f"M{front_month}" for x in s.columns})
    s.attrs = {}
    return s


def generate_structure_series(contracts: pd.DataFrame,
                              mx: int = 1,
                              my: int = 2,
                              roll_days: int = 0,
                              mx_df: pd.DataFrame = None,
                              my_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Given a list of contracts, calculate structure.
    """

    colname = f"M{mx}-M{my}"

    if mx_df is not None and my_df is not None:
        df = pd.concat([mx_df, my_df], axis=1)
    else:
        df = generate_multiple_continuous_series(contracts, months=[mx, my], roll_days=roll_days)

    df[colname] = df[f"M{mx}"].sub(df[f"M{my}"]).dropna()
    df = df[[colname]]
    df = df.dropna(how="all", axis="rows")
    return df
