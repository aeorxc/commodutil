import datetime
import re

import pandas as pd



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
