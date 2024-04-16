import os
import pytest
from commodutil.forward import fly
import pandas as pd


def test_fly():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    cl = pd.read_csv(
        os.path.join(dirname, "../test_cl.csv"),
        index_col=0,
        parse_dates=True,
        dayfirst=True,
    )

    res = fly.fly(cl, m1=1, m2=2, m3=3)
    assert res["JanFebMar 2020"].loc[pd.to_datetime("2019-01-03")] == pytest.approx(-0.02, abs=1e-2)
    assert res["JanFebMar 2021"].loc[pd.to_datetime("2019-05-21")] == pytest.approx(0.02, abs=1e-2)


def test_fly2():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    cl = pd.read_csv(
        os.path.join(dirname, "../test_cl.csv"),
        index_col=0,
        parse_dates=True,
        dayfirst=True,
    )

    res = fly.fly(cl, m1=12, m2=1, m3=3)
    assert res["DecJanMar 2020"].loc[pd.to_datetime("2019-01-03")] == pytest.approx(0.06, abs=1e-2)
    assert res["DecJanMar 2021"].loc[pd.to_datetime("2019-05-21")] == pytest.approx(-0.14, abs=1e-2)
