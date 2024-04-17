import pandas as pd
import pytest
import os
from commodutil.forward import calendar


def test_calendar():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    cl = pd.read_csv(
        os.path.join(dirname, "../test_cl.csv"),
        index_col=0,
        parse_dates=True,
        dayfirst=True,
    )

    res = calendar.cal_contracts(cl)
    assert res is not None
    assert res["CAL 2020"]["2020-01-02"] == pytest.approx(59.174, abs=1e-3)


def test_calendar_spread():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    cl = pd.read_csv(
        os.path.join(dirname, "../test_cl.csv"),
        index_col=0,
        parse_dates=True,
        dayfirst=True,
    )

    res = calendar.cal_spreads(calendar.cal_contracts(cl))
    assert res["CAL 2020-2021"]["2020-01-02"] == pytest.approx(4.35, abs=1e-2)


def test_half_year():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    cl = pd.read_csv(
        os.path.join(dirname, "../test_cl.csv"),
        index_col=0,
        parse_dates=True,
        dayfirst=True,
    )

    res = calendar.half_year_contracts(cl)
    assert res["H1 2020"]["2019-01-02"] == pytest.approx(50.04, abs=1e-2)


def test_half_year_spread():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    cl = pd.read_csv(
        os.path.join(dirname, "../test_cl.csv"),
        index_col=0,
        parse_dates=True,
        dayfirst=True,
    )

    res = calendar.half_year_spreads(calendar.half_year_contracts(cl))
    assert res["H1H2 2020"]["2019-01-02"] == pytest.approx(-0.578, abs=1e-2)