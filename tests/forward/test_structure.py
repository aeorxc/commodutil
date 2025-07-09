import pandas as pd
import pytest

from commodutil.forward.util import convert_contract_to_date
from commodutil.forward.structure import generate_structure_series


def test_continuous_futures(contracts):
    cl = contracts.rename(
        columns={
            x: pd.to_datetime(convert_contract_to_date(x))
            for x in contracts.columns
        }
    )

    res = generate_structure_series(cl, mx=1, my=2)
    assert res["M1-M2"].loc[pd.to_datetime("2020-11-20")] == pytest.approx(-0.27, abs=0.01)
