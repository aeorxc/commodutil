import pandas as pd
from commodutil.pandasutil import apply_formula


def calc(df: pd.DataFrame, dest: str, orig: str, freight: str = None) -> pd.DataFrame:
    """Calculate an arb by pulling all load and discharge symbols and subtract freight
    """

    formula = f"{dest} - {orig}"
    if freight is not None:
        formula = f"{formula} - {freight}"

    df = apply_formula(df, formula)
    return df
