import os
import pytest
from commodutil.forward import util
import pandas as pd


def test_convert_columns_to_date():
    df = pd.DataFrame([], columns=["2020J", "2020M", "Test"])
    res = util.convert_columns_to_date(df)
    assert pd.to_datetime("2020-04-1") in res.columns


def test_convert_contract_to_date():
    res = util.convert_contract_to_date("2020F")
    assert res == "2020-1-1"