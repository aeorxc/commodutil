import pandas as pd
import pytest

from commodutil.forward.util import convert_contract_to_date
from commodutil.forward.continuous import generate_series


def test_continuous_futures(contracts):
    cl = contracts.rename(
        columns={
            x: pd.to_datetime(convert_contract_to_date(x))
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

    res = generate_series(cl, expiry_dates=expiry_dates, front_month=[1, 2])
    assert res["M1"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.15, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.06, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(41.90, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.42, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.28, abs=0.01)

    res = generate_series(cl, expiry_dates=expiry_dates, roll_days=1)
    assert res["M1"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(41.74, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.42, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.06, abs=0.01)

    res = generate_series(
        cl, expiry_dates=expiry_dates, front_month=2, roll_days=1
    )
    assert res["M2"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(41.90, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.64, abs=0.01)
    assert res["M2"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.28, abs=0.01)

    res = generate_series(
        cl, expiry_dates=expiry_dates, front_month=1, back_adjust=True
    )
    assert res["M1"].loc[pd.to_datetime("2020-11-19")] == pytest.approx(42.01, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(42.42, abs=0.01)
    assert res["M1"].loc[pd.to_datetime("2020-11-23")] == pytest.approx(43.06, abs=0.01)
