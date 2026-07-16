import warnings

import pytest
from commodutil.forward import calendar


def test_calendar(cl):
    res = calendar.cal_contracts(cl)
    assert res is not None
    assert res["CAL 2020"]["2020-01-02"] == pytest.approx(59.174, abs=1e-3)

def test_calendar_spread(cl):
    res = calendar.cal_spreads(calendar.cal_contracts(cl))
    assert res["CAL 2020-2021"]["2020-01-02"] == pytest.approx(4.35, abs=1e-2)

def test_half_year(cl):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = calendar.half_year_contracts(cl)
    assert res["H1 2020"]["2019-01-02"] == pytest.approx(50.04, abs=1e-2)

def test_half_year_spread(cl):
    res = calendar.half_year_spreads(calendar.half_year_contracts(cl))
    assert res["H1H2 2020"]["2019-01-02"] == pytest.approx(-0.578, abs=1e-2)
