import pandas as pd
import pytest

from commodutil import convfactors


class TestUtils:
    def test_conv_factor(self):
        # Test diesel kt to km3 conversion
        # 1 kt diesel = 1,000,000 kg / 843 kg/m³ = 1,186 m³ = 0.000001186 km³
        res = convfactors.convfactor("kt", "km3", "diesel")
        assert res * 1e6 == pytest.approx(1.184455, abs=1e-3)  # Convert to m³ for comparison

        # Test alias support (ulsd = diesel)
        res = convfactors.convfactor("kt", "km3", "ulsd")
        assert res * 1e6 == pytest.approx(1.184455, abs=1e-3)  # Convert to m³ for comparison

        # Test reverse conversion
        res = convfactors.convfactor("km3", "kt", "ulsd")
        assert res / 1e6 == pytest.approx(0.844269902, abs=1e-3)  # Scale for comparison

    def test_convert(self):
        diesel_kt = 520

        # Test kt to km3 conversion
        res = convfactors.convert(diesel_kt, "kt", "km3", "diesel")
        # 520 kt = 520 * 1,186 m³ = 616,840 m³ = 0.00061684 km³
        assert res * 1e6 == pytest.approx(615.9168, abs=1e-2)  # Convert to m³ for comparison
        
        # Test kt to bbl conversion
        res = convfactors.convert(100, "kt", "bbl", "diesel")
        # Using density-based calculation: ~746,122 bbl
        assert res == pytest.approx(745000, rel=1e-3)  # Round to hundreds

    def test_convert_series(self):
        # Test pandas Series with daily rates
        d = pd.Series(
            [50, 60, 70, 80], index=pd.date_range("2020", periods=4, freq="MS")
        )
        
        # Convert kt/month to bbl/day
        res = convfactors.convert(d, "kt/month", "bbl/day", "diesel")
        # Expected: 50 kt/month for Jan 2020 (31 days) = 50*7461/31 = ~12034 bbl/d
        assert res["2020-01"].iloc[0] == pytest.approx(12016.129, abs=1)  # Round to units
        
        # Convert bbl/day to kt/month
        res = convfactors.convert(d, "bbl/day", "kt/month", "diesel")
        # Expected: 50 bbl/d for Jan 2020 (31 days) = 50*31/7461 = ~0.208 kt
        assert res["2020-01"].iloc[0] == pytest.approx(0.208054, abs=1e-3)

    def test_energy_conversions(self):
        """Test energy to volume/mass conversions"""
        # Volume to energy (diesel has 36.624 GJ/m³)
        res = convfactors.convert(1, "m^3", "GJ", "diesel")
        assert res == pytest.approx(36.624, abs=0.01)
        
        # Energy to volume
        res = convfactors.convert(36.624, "GJ", "m^3", "diesel")
        assert res == pytest.approx(1.0, abs=0.01)
        
        # Mass to energy (diesel: 0.844 kg/L, 36.624 GJ/m³)
        # 1 mt = 1.184 m³ = 43.38 GJ
        res = convfactors.convert(1, "mt", "GJ", "diesel")
        assert res == pytest.approx(43.38, abs=0.01)
        
        # Energy to mass
        res = convfactors.convert(43.38, "GJ", "mt", "diesel")
        assert res == pytest.approx(1.0, abs=0.01)
        
        # Test commodity without energy content raises error
        with pytest.raises(ValueError, match="No energy content"):
            convfactors.convert(1, "m^3", "GJ", "crude")
        
        # Test gaseous natural gas energy conversion
        res = convfactors.convert(1000, "m^3", "GJ", "natural_gas")
        assert res == pytest.approx(36.0, abs=0.001)  # 1000 m³ * 0.036 GJ/m³ = 36 GJ
    
    def test_simple_conversions(self):
        """Test simple unit conversions that don't need commodity"""
        # Barrel to gallons
        res = convfactors.convert(1, "bbl", "gal")
        assert res == pytest.approx(42, abs=1)
        
        # Barrel to liters  
        res = convfactors.convert(1, "bbl", "L")
        assert res == pytest.approx(158.987, abs=1e-2)
        
        # Kiloton to metric ton
        res = convfactors.convert(1, "kt", "mt")
        assert res == pytest.approx(1000, abs=1)
    
    def test_commodity_conversions(self):
        """Test conversions that require commodity context"""
        # Diesel conversions
        res = convfactors.convfactor("kt", "bbl", "diesel")
        # Diesel: 0.843 kg/L means 1 kt = 1,000,000/0.843 L = 7461 bbl
        assert res == pytest.approx(7450, abs=1)  # ~7461 bbl per kt
        
        # Gasoline conversions (lighter than diesel, more bbl per kt)
        res = convfactors.convfactor("kt", "bbl", "gasoline")
        # Gasoline is lighter, so more barrels per kt
        assert res  > 7450  # Should be > diesel
        
        # Jet fuel conversions
        res = convfactors.convfactor("kt", "bbl", "jet")
        assert res == pytest.approx(7880, abs=1)  # ~7,765 bbl per kt
        
        # Heavy fuel oil (heavier than diesel, fewer bbl per kt)
        res = convfactors.convfactor("kt", "bbl", "fuel_oil")
        assert res  < 7450  # Should be < diesel
    
    def test_aliases(self):
        """Test commodity aliases work correctly"""
        # ULSD = diesel
        ulsd_factor = convfactors.convfactor("kt", "bbl", "ulsd")
        diesel_factor = convfactors.convfactor("kt", "bbl", "diesel")
        assert ulsd_factor == diesel_factor
        
        # GO (gasoil) = diesel
        go_factor = convfactors.convfactor("kt", "bbl", "go")
        assert go_factor == diesel_factor
        
        # Gas = gasoline
        gas_factor = convfactors.convfactor("kt", "bbl", "gas")
        gasoline_factor = convfactors.convfactor("kt", "bbl", "gasoline")
        assert gas_factor == gasoline_factor
    
    def test_error_handling(self):
        """Test appropriate errors are raised"""
        # Should fail without commodity for mass-volume conversion
        with pytest.raises(ValueError):
            convfactors.convert(100, "kt", "bbl")
        
        # Should fail for unknown commodity
        with pytest.raises(ValueError):
            convfactors.convfactor("kt", "bbl", "unknown_commodity")
        
        # Should fail for incompatible dimensions without commodity
        with pytest.raises(ValueError):
            convfactors.convert(100, "bbl", "kg")
    
    def test_volume_mass_conversions(self):
        """Test volume to mass conversions"""
        # 1000 bbl of diesel to mt
        res = convfactors.convert(1000, "bbl", "mt", "diesel")
        # 1000 bbl / 7450 bbl/kt = 0.134 kt = 134 mt
        assert res == pytest.approx(134, abs=1)
        
        # 1000 m3 of gasoline to kt
        res = convfactors.convert(1000, "m^3", "kt", "gasoline")
        # 1000 m3 * 0.745 kg/L * 1000 L/m3 / 1e6 kg/kt = 0.755079 kt
        assert res == pytest.approx(0.755079, abs=1e-3)
    
    def test_rate_conversions(self):
        """Test rate conversions (daily/monthly/yearly)"""
        # Day to month conversions (using average 30.4375 days/month)
        res = convfactors.convert(100, "bbl/day", "bbl/month", "diesel")
        assert res == pytest.approx(3043.75, abs=0.1)  # 100 * 30.4375
        
        # Month to day conversions
        res = convfactors.convert(3043.75, "bbl/month", "bbl/day", "diesel")
        assert res == pytest.approx(100, abs=0.01)
        
        # Year to day conversions (365.25 days/year)
        res = convfactors.convert(36525, "bbl/year", "bbl/day")
        assert res == pytest.approx(100, abs=0.01)
        
        # Combined unit and rate conversion
        res = convfactors.convert(1, "kt/month", "bbl/day", "diesel")
        assert res == pytest.approx(244.77, abs=0.01)  # 7450 bbl/kt / 30.4375 days/month
    
    def test_normalize_unit(self):
        """Test unit normalization"""
        from commodutil.convfactors import converter
        
        # Test cubic meter variations
        assert converter._normalize_unit("m³") == "m^3"
        assert converter._normalize_unit("m**3") == "m^3"
        assert converter._normalize_unit("cubic_meter") == "m^3"
        assert converter._normalize_unit("CUBIC_METER") == "m^3"
        
        # Test energy unit normalization
        assert converter._normalize_unit("BTU") == "Btu"
        assert converter._normalize_unit("MMBTU") == "MMBtu"
        
        # Test whitespace handling
        assert converter._normalize_unit("  bbl  ") == "bbl"
    
    def test_commodity_properties(self):
        """Test commodity property access and helper functions"""
        # Test list_commodities
        commodities = convfactors.list_commodities()
        assert "diesel" in commodities
        assert "gasoline" in commodities
        assert "jet" in commodities
        assert "crude" in commodities
        assert len(commodities) > 10
        
        # Test list_units
        units = convfactors.list_units()
        assert "bbl" in units
        assert "kt" in units
        assert "GJ" in units
        assert "bbl/day" in units
        
        # Test getting commodity with alias
        from commodutil.convfactors import converter
        diesel = converter.get_commodity("diesel")
        ulsd = converter.get_commodity("ulsd")
        assert diesel.name == ulsd.name
        
        # Test unknown commodity raises error
        with pytest.raises(ValueError, match="Unknown commodity"):
            converter.get_commodity("unknown_fuel")
    
    def test_edge_cases(self):
        """Test edge cases and special scenarios"""
        # Test zero values
        res = convfactors.convert(0, "kt", "bbl", "diesel")
        assert res == 0
        
        # Test negative values (should work for temperature differences, etc)
        res = convfactors.convert(-100, "bbl", "L")
        assert res == pytest.approx(-15898.7, abs=0.1)
        
        # Test very large numbers
        res = convfactors.convert(1e6, "kt", "bbl", "diesel")
        assert res == pytest.approx(7.45e9, rel=1e-3)
        
        # Test very small numbers (0.001 mt = 1 kg)
        res = convfactors.convert(0.001, "mt", "bbl", "diesel")
        assert res == pytest.approx(0.00745, abs=0.0001)  # 0.001 mt * 7450 bbl/mt
    
    def test_natural_gas_special_cases(self):
        """Test natural gas conversions (special density handling)"""
        # Natural gas has 0.0 kg/L density (gaseous state)
        # Should raise error for mass conversions
        with pytest.raises(ValueError):
            convfactors.convert(1, "m^3", "kg", "natural_gas")
        
        # But energy conversions should work
        res = convfactors.convert(1e9, "m^3", "GJ", "natural_gas")
        assert res == pytest.approx(36000000, abs=1000)  # 1 BCM * 0.036 GJ/m³ = 36 million GJ = 0.036 PJ
    
    def test_unit_dimension_checks(self):
        """Test internal dimension checking methods"""
        from commodutil.convfactors import converter
        
        # Test _is_energy
        assert converter._is_energy("GJ") == True
        assert converter._is_energy("bbl") == False
        assert converter._is_energy("toe") == True
        
        # Test _is_mass
        assert converter._is_mass("kt") == True
        assert converter._is_mass("bbl") == False
        assert converter._is_mass("mt") == True
        
        # Test _is_volume
        assert converter._is_volume("bbl") == True
        assert converter._is_volume("m^3") == True
        assert converter._is_volume("kt") == False
    
    def test_incompatible_conversions(self):
        """Test that incompatible conversions raise appropriate errors"""
        # Length to mass should fail
        with pytest.raises(ValueError):
            convfactors.convert(100, "meter", "kg")
        
        # Energy to volume without commodity should fail
        with pytest.raises(ValueError, match="Commodity required"):
            convfactors.convert(100, "GJ", "bbl")
        
        # Mass to volume without commodity should fail  
        with pytest.raises(ValueError, match="Commodity required"):
            convfactors.convert(100, "kt", "m^3")
    
    def test_series_with_different_periods(self):
        """Test pandas Series with different period conversions"""
        # Create series with yearly data
        dates = pd.date_range('2024', periods=3, freq='YS')
        series = pd.Series([365250, 400000, 380000], index=dates)
        
        # Year to day conversion
        result = convfactors.convert(series, 'bbl/year', 'bbl/day')
        assert result.iloc[0] == pytest.approx(1000, abs=1)
        
        # Test series without calendar-aware index (no days_in_month attribute)
        simple_series = pd.Series([100, 110, 105])
        result = convfactors.convert(simple_series, 'bbl/day', 'bbl/month')
        assert result.iloc[0] == pytest.approx(3043.75, abs=0.1)
    
    def test_caching_and_properties(self):
        """Test caching and property methods"""
        from commodutil.convfactors import converter
        
        # Test get_commodity caching (should use same object)
        diesel1 = converter.get_commodity("diesel")
        diesel2 = converter.get_commodity("diesel")
        assert diesel1 is diesel2  # Same cached object
        
        # Test available_commodities property
        commodities = converter.available_commodities
        assert isinstance(commodities, list)
        assert "diesel" in commodities
        
        # Test available_units property
        units = converter.available_units
        assert isinstance(units, list)
        assert "bbl/day" in units




