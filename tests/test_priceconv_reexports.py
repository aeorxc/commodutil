"""Compatibility checks for the price conversion module split."""


def test_priceconv_moved_names_reexported_from_convfactors():
    from commodutil import convfactors
    from commodutil import priceconv

    moved_names = (
        "align_fx",
        "convert_price",
        "convert_currency_leg",
        "ConversionResult",
        "convert_price_result",
    )
    for name in moved_names:
        assert getattr(convfactors, name) is getattr(priceconv, name)
