import os
import pytest
from commodutil.forward import spreads
import pandas as pd


def test_timespreads(cl):
    contracts = cl
    res = spreads.time_spreads_monthly(contracts, m1=6, m2=12)
    assert res['JunDec 2019'].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-1.51, abs=1e-2)
    assert res['JunDec 2019'].loc[pd.to_datetime("2019-05-21")] == pytest.approx(0.37, abs=1e-2)

    res = spreads.time_spreads(contracts, m1=12, m2=12)
    assert res['DecDec 2019'].loc[pd.to_datetime("2019-11-20")] == pytest.approx(3.56, abs=1e-2)
    assert res['DecDec 2020'].loc[pd.to_datetime("2019-03-20")] == pytest.approx(2.11, abs=1e-2)


def test_all_monthly_spreads(cl):
    res = spreads.all_monthly_spreads(cl, start_date=pd.to_datetime("2020-01-01"),
                                      end_date=pd.to_datetime("2023-12-31"))

    # Add your assertions here based on what you expect the output to be
    # For example:
    assert res["JanFeb 2020"].loc[pd.to_datetime("2019-01-03")] == pytest.approx(-0.10, abs=1e-2)
    assert res["JanFeb 2021"].loc[pd.to_datetime("2019-05-21")] == pytest.approx(0.31, abs=1e-2)
