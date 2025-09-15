import pandas as pd
import pytest

from commodutil import convfactors

def test_convert_price_scalar_mt_to_bbl_gasoline():
    # 100 $/mt gasoline ˜ 12 $/bbl
    out = convfactors.convert_price(100.0, 'mt', 'bbl', 'gasoline')
    assert out == pytest.approx(12.0, rel=1e-2)


def test_convert_price_scalar_gal_to_bbl():
    # 2.5 $/gal ? $/bbl (×42)
    out = convfactors.convert_price(2.5, 'gal', 'bbl')
    assert out == pytest.approx(105.0, rel=1e-6)


def test_convert_price_series_diesel():
    s = pd.Series([700.0, 750.0, 800.0])  # $/mt
    out = convfactors.convert_price(s, 'mt', 'bbl', 'diesel')
    # Diesel mt?bbl factor ~7.45.. -> prices scale down accordingly
    ratios = (out / s)
    # Use convfactor as expected ratio
    expected = 1.0 / convfactors.convfactor('mt', 'bbl', 'diesel')
    assert ratios.median() == pytest.approx(expected, rel=1e-6)
