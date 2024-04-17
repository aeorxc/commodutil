# test_forwards.py
import pytest
from commodutil import forwards
import pandas as pd


def test_timespreads(contracts):
    res = forwards.time_spreads(contracts, m1=6, m2=12)
    assert res[2019].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-1.51, abs=0.01)
    assert res[2019].loc[pd.to_datetime("2019-05-21")] == pytest.approx(0.37, abs=0.01)

    res = forwards.time_spreads(contracts, m1=12, m2=12)
    assert res[2019].loc[pd.to_datetime("2019-11-20")] == pytest.approx(3.56, abs=0.01)
    assert res[2020].loc[pd.to_datetime("2019-03-20")] == pytest.approx(2.11, abs=0.01)


def test_timespreads_quaterly(contracts):
    res = forwards.time_spreads(contracts, m1="Q1", m2="Q2")
    assert res[2020].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.33, abs=0.01)
    assert res[2020].loc[pd.to_datetime("2019-05-21")] == pytest.approx(1.05,abs=0.01)

    res = forwards.time_spreads(contracts, m1="Q4", m2="Q1")
    assert res[2020].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.25,abs=0.01)
    assert res[2020].loc[pd.to_datetime("2019-05-21")] == pytest.approx(0.91,abs=0.01)


def test_all_spread_combinations(contracts):
    res = forwards.all_spread_combinations(contracts)
    assert "Q1" in res
    assert "Q1Q2" in res
    assert "Calendar" in res
    assert "JanFeb" in res
    assert "JanFebMar" in res


def test_spread_combination_calendar(contracts):
    res = forwards.spread_combination(contracts, "calendar")
    assert res is not None
    assert res[2020].loc[pd.to_datetime("2020-01-02")] == pytest.approx(59.174,abs=0.01)


def test_spread_combination_calendar_spread(contracts):
    res = forwards.spread_combination(contracts, "calendar spread")
    assert res["CAL 2020-2021"].loc[pd.to_datetime("2020-01-02")] == pytest.approx(4.35, abs=0.01)


def test_spread_combination_half_year(contracts):
    res = forwards.spread_combination(contracts, "half year")
    assert res["H1 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(50.04,abs=0.01)


def test_spread_combination_half_year_spread(contracts):
    res = forwards.spread_combination(contracts, "half year spread")
    assert res["H1H2 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.578, abs=0.01)


def test_spread_combination_quarter(contracts):
    res = forwards.spread_combination(contracts, "q1")
    assert res["Q1 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(49.88, abs=0.01)


def test_spread_combination_quarter_spread(contracts):
    res = forwards.spread_combination(contracts, "q1q2")
    assert res["Q1Q2 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.33,abs=0.01)

    res = forwards.spread_combination(contracts, "q1q3")
    assert res["Q1Q3 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.58, abs=0.01)


def test_spread_combination_monthly(contracts):
    res = forwards.spread_combination(contracts, "monthly", col_format="%b%b %y")

    assert res["JanFeb 20"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.11, abs=0.01)
    assert res["FebMar 21"].loc[pd.to_datetime("2020-01-02")] == pytest.approx(0.35, abs=0.01)


def test_spread_combination_month(contracts):
    res = forwards.spread_combination(contracts, "jan")
    assert res["Jan 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(49.77, abs=0.01)


def test_spread_combination_month_spread_janfeb(contracts):
    res = forwards.spread_combination(contracts, "janfeb")
    assert res["JanFeb 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.11, abs=0.01)


def test_spread_combination_month_spread_decjan(contracts):
    res = forwards.spread_combination(contracts, "decjan")
    assert res["DecJan 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.06, abs=0.01)


def test_spread_combination_month_fly(contracts):
    res = forwards.spread_combination(contracts, "janfebmar")
    assert res["JanFebMar 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(0.0,abs=0.01)


def test_spread_combination_quarter_fly(contracts):
    res = forwards.spread_combination(contracts, "q4q1q2")
    assert res["Q4Q1Q2 2020"].loc[pd.to_datetime("2019-01-02")] == pytest.approx(-0.023, abs=0.01)


def test_continuous_futures(contracts):
    cl = contracts.rename(
        columns={
            x: pd.to_datetime(forwards.convert_contract_to_date(x))
            for x in contracts.columns
        }
    )

    expiry_dates = {
        "2019-01-01": "2018-12-19",
        "2019-02-01": "2019-01-22",
        "2019-03-01": "2019-02-20",
        "2019-04-01": "2019-03-20",
        "2019-05-01": "2019-04-22",
        "2019-06-01": "2019-05-21",
        "2019-07-01": "2019-06-20",
        "2019-08-01": "2019-07-22",
        "2019-09-01": "2019-08-20",
        "2019-10-01": "2019-09-20",
        "2019-11-01": "2019-10-22",
        "2019-12-01": "2019-11-20",
        "2020-01-01": "2019-12-19",
        "2020-02-01": "2020-01-21",
        "2020-03-01": "2020-02-20",
        "2020-04-01": "2020-03-20",
        "2020-05-01": "2020-04-21",
        "2020-06-01": "2020-05-19",
        "2020-07-01": "2020-06-22",
        "2020-08-01": "2020-07-21",
        "2020-09-01": "2020-08-20",
        "2020-10-01": "2020-09-22",
        "2020-11-01": "2020-10-20",
        "2020-12-01": "2020-11-20",
        "2021-01-01": "2021-01-20",
    }

    res = forwards.continuous_futures(cl, expiry_dates=expiry_dates, front_month=[1,2])
    assert res["M1"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.15, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.06, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(41.90, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.42, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.28, abs=0.01)

    res = forwards.continuous_futures(cl, expiry_dates=expiry_dates, roll_days=1)
    assert res["M1"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(41.74, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.42, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.06, abs=0.01)

    res = forwards.continuous_futures(
        cl, expiry_dates=expiry_dates, front_month=2, roll_days=1
    )
    assert res["M2"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(41.90, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.64, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.28, abs=0.01)

    res = forwards.continuous_futures(
        cl, expiry_dates=expiry_dates, front_month=1, back_adjust=True
    )
    assert res["M1"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(42.01, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.42, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.06, abs=0.01)