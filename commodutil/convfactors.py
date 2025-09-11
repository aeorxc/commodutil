"""
Modern implementation of commodity unit conversions using pint
No backwards compatibility constraints - clean slate design
"""

import pint
from typing import Union, Optional
from dataclasses import dataclass
import pandas as pd
from functools import lru_cache

# Initialize pint with custom definitions
ureg = pint.UnitRegistry()

# Define oil & gas specific units
ureg.define('barrel = 158.987294928 liter = bbl')
ureg.define('gallon = 3.785411784 liter = gal')
ureg.define('metric_ton = 1000 kilogram = mt')
ureg.define('kiloton = 1000 metric_ton = kt')
ureg.define('cubic_kilometer = 1e9 meter**3 = km3')  # 1 km³ = 1 billion m³
ureg.define('gigajoule = 1e9 joule = gj = GJ')

@dataclass
class Commodity:
    """Represents a commodity with its physical properties"""
    name: str
    density: pint.Quantity  # kg/L or API gravity
    energy_content: Optional[pint.Quantity] = None  # GJ/m³ or similar
    
    def __post_init__(self):
        # Ensure quantities have correct dimensions
        if not isinstance(self.density, pint.Quantity):
            self.density = self.density * ureg.kg / ureg.liter
        if self.energy_content and not isinstance(self.energy_content, pint.Quantity):
            self.energy_content = self.energy_content * ureg.GJ / ureg.m**3

# Define commodities with their properties and correct industry factors
COMMODITIES = {
    # Crude oil (BP approximate conversion factors)
    # 1 mt ≈ 7.33 bbl and ≈ 1.165 kL => density ≈ 0.85809 kg/L
    'crude': Commodity('crude', 0.85809151 * ureg.kg/ureg.L, None),

    # Light ends - tuned to match kbbl/kt figures exactly
    'gasoline': Commodity('gasoline', 0.755079324 * ureg.kg/ureg.L, 33.7898 * ureg.GJ/ureg.m**3),  # BP: 44.75 GJ/t
    'naphtha': Commodity('naphtha', 0.706720311 * ureg.kg/ureg.L, None),  # 8.90 kbbl/kt
    'ethanol': Commodity('ethanol', 0.755079324 * ureg.kg/ureg.L, 21 * ureg.GJ/ureg.m**3),  # 8.33 kbbl/kt
    
    # Middle distillates  
    'diesel': Commodity('diesel', 0.844269902 * ureg.kg/ureg.L, 36.624428 * ureg.GJ/ureg.m**3),  # BP: 43.38 GJ/t
    'jet': Commodity('jet', 0.798199336 * ureg.kg/ureg.L, 35.056915 * ureg.GJ/ureg.m**3),  # BP: 43.92 GJ/t
    'fame': Commodity('fame', 0.892001564 * ureg.kg/ureg.L, 33 * ureg.GJ/ureg.m**3),  # 7.051345 kbbl/kt
    'hvo': Commodity('hvo', 0.781731391 * ureg.kg/ureg.L, 34 * ureg.GJ/ureg.m**3),  # 8.046 kbbl/kt
    
    # Heavy products
    'vgo': Commodity('vgo', 0.911566778 * ureg.kg/ureg.L, None),  # 6.90 kbbl/kt
    'fuel_oil': Commodity('fuel_oil', 0.990521381 * ureg.kg/ureg.L, 41.175974 * ureg.GJ/ureg.m**3),  # BP: 41.57 GJ/t
    
    # LPG and Natural gas (liquefied)
    'lpg': Commodity('lpg', 0.541 * ureg.kg/ureg.L, 24.96715 * ureg.GJ/ureg.m**3),  # BP: LPG 46.15 GJ/t
    'natgas': Commodity('natgas', 0.542225066 * ureg.kg/ureg.L, 26.137 * ureg.GJ/ureg.m**3),  # LNG figures
    
    # Light gases
    'ethane': Commodity('ethane', 0.373 * ureg.kg/ureg.L, 18.4262 * ureg.GJ/ureg.m**3),  # BP: 49.4 GJ/t
    
    # BP product basket (optional reference)
    'product_basket': Commodity('product_basket', 0.781 * ureg.kg/ureg.L, 33.642356 * ureg.GJ/ureg.m**3),
}

# Aliases for compatibility
ALIASES = {
    'ulsd': 'diesel',
    'gasoil': 'diesel', 
    'go': 'diesel',
    'gas': 'gasoline',
    'mogas': 'gasoline',
    'fueloil': 'fuel_oil',
    'fo': 'fuel_oil',
    'lng': 'natgas',
    'kerosene': 'jet',
    'propane': 'lpg',
}

class CommodityConverter:
    """Clean, modern interface for commodity unit conversions"""
    
    def __init__(self):
        self.ureg = ureg
        self.commodities = COMMODITIES
        self.aliases = ALIASES
    
    @lru_cache(maxsize=128)
    def get_commodity(self, name: str) -> Commodity:
        """Get commodity object, resolving aliases"""
        name = name.lower()
        name = self.aliases.get(name, name)
        if name not in self.commodities:
            raise ValueError(f"Unknown commodity: {name}")
        return self.commodities[name]
    
    def convert(self, 
                value: Union[float, pd.Series],
                from_unit: str,
                to_unit: str, 
                commodity: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Convert between units, using commodity properties when needed
        
        Examples:
            # Simple unit conversion (no commodity needed)
            convert(100, 'bbl', 'L')  
            
            # Mass to volume (needs commodity density)
            convert(100, 'kt', 'bbl', commodity='diesel')
            
            # Energy conversions
            convert(1000, 'm³', 'GJ', commodity='diesel')
            
            # With pandas Series and daily rates
            convert(series, 'kt/month', 'bbl/day', commodity='gasoline')
        """
        # Parse units to handle daily/monthly rates
        from_rate = self._parse_rate_unit(from_unit)
        to_rate = self._parse_rate_unit(to_unit)
        
        # Get base units
        from_base = from_rate['base']
        to_base = to_rate['base']
        
        # Create quantity
        if isinstance(value, pd.Series):
            result = self._convert_series(value, from_base, to_base, commodity,
                                        from_rate['period'], to_rate['period'])
        else:
            result = self._convert_scalar(value, from_base, to_base, commodity)
        
        return result
    
    def _convert_scalar(self, value: float, from_unit: str, to_unit: str, 
                       commodity: Optional[str]) -> float:
        """Convert a scalar value"""
        qty = value * self.ureg(from_unit)
        
        # Get dimensions
        from_dim = self.ureg.get_dimensionality(from_unit)
        to_dim = self.ureg.get_dimensionality(to_unit)
        
        # Check if we need commodity context
        if from_dim == to_dim:
            # Simple unit conversion
            return qty.to(to_unit).magnitude
        
        # Energy conversions (check first as they may involve volume/mass)
        if self._needs_energy(from_dim, to_dim):
            if not commodity:
                raise ValueError(f"Commodity required for energy conversion")
            
            comm = self.get_commodity(commodity)
            if not comm.energy_content:
                raise ValueError(f"No energy content defined for {commodity}")
            
            if '[energy]' in str(from_dim):
                # Energy to volume
                energy_J = qty.to('J')
                volume_m3 = energy_J / comm.energy_content
                return volume_m3.to(to_unit).magnitude
            else:
                # Volume/mass to energy
                # First convert to volume if needed
                if '[mass]' in str(from_dim):
                    mass_kg = qty.to('kg')
                    volume_m3 = mass_kg / comm.density.to('kg/m^3')
                else:
                    volume_m3 = qty.to('m^3')
                energy = volume_m3 * comm.energy_content
                return energy.to(to_unit).magnitude
        
        # Mass to volume or vice versa needs density
        elif self._needs_density(from_dim, to_dim):
            if not commodity:
                raise ValueError(f"Commodity required for {from_unit} to {to_unit}")
            
            comm = self.get_commodity(commodity)
            
            if '[mass]' in str(from_dim) and '[length] ** 3' in str(to_dim):
                # Mass to volume: Use industry standard conversions
                # For kt to bbl: use the fact that density tells us kg/L
                # 1 kt = 1,000,000 kg
                # density kg/L means: 1 L = density kg
                # So: 1 kg = 1/density L
                # Therefore: 1 kt = 1,000,000/density L = 1,000,000/density/158.987 bbl
                mass_kg = qty.to('kg')
                volume_L = mass_kg / comm.density
                return volume_L.to(to_unit).magnitude
            elif '[length] ** 3' in str(from_dim) and '[mass]' in str(to_dim):
                # Volume to mass: convert volume to L first, then multiply by density
                volume_L = qty.to('L')
                mass_kg = volume_L * comm.density
                return mass_kg.to(to_unit).magnitude
        
        else:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit} - incompatible dimensions")
    
    def _convert_series(self, series: pd.Series, from_unit: str, to_unit: str,
                       commodity: Optional[str], from_period: Optional[str],
                       to_period: Optional[str]) -> pd.Series:
        """Convert a pandas Series with optional rate handling"""
        result = series.copy()
        
        # Handle period conversions for rates
        if from_period == 'day' and from_period != to_period:
            # Daily to monthly/yearly
            if hasattr(series.index, 'days_in_month'):
                result = result * series.index.days_in_month
        elif to_period == 'day' and from_period != to_period:
            # Monthly/yearly to daily
            if hasattr(series.index, 'days_in_month'):
                result = result / series.index.days_in_month
        
        # Apply unit conversion
        factor = self._convert_scalar(1.0, from_unit, to_unit, commodity)
        result = result * factor
        
        return result
    
    def _parse_rate_unit(self, unit: str) -> dict:
        """Parse units like 'bbl/day' or 'kt/month'"""
        if '/' in unit:
            base, period = unit.split('/')
            period = period.rstrip('s')  # Remove plural
            return {'base': base, 'period': period}
        return {'base': unit, 'period': None}
    
    def _needs_density(self, from_dim, to_dim) -> bool:
        """Check if conversion needs density"""
        from_str = str(from_dim)
        to_str = str(to_dim)
        mass_to_vol = '[mass]' in from_str and '[length] ** 3' in to_str
        vol_to_mass = '[length] ** 3' in from_str and '[mass]' in to_str
        return mass_to_vol or vol_to_mass
    
    def _needs_energy(self, from_dim, to_dim) -> bool:
        """Check if conversion needs energy content"""
        return '[energy]' in str(from_dim) or '[energy]' in str(to_dim)
    
    @property
    def available_commodities(self) -> list:
        """List all available commodities"""
        return list(self.commodities.keys())
    
    @property 
    def available_units(self) -> list:
        """List common units for oil & gas"""
        return [
            # Volume
            'bbl', 'barrel', 'L', 'liter', 'm³', 'cubic_meter', 'gal', 'gallon',
            # Mass
            'kg', 'mt', 'metric_ton', 'kt', 'kiloton', 't', 'tonne',
            # Energy
            'J', 'GJ', 'gigajoule', 'MJ', 'megajoule', 'BTU', 'MMBTU',
            # Rates
            'bbl/day', 'kt/month', 'm³/day', 'mt/year'
        ]

# Global converter instance for convenience
converter = CommodityConverter()

# Convenience functions for direct use
def convert(value, from_unit: str, to_unit: str, commodity: Optional[str] = None):
    """Convert values between units"""
    return converter.convert(value, from_unit, to_unit, commodity)

def convfactor(from_unit: str, to_unit: str, commodity: Optional[str] = None) -> float:
    """Get conversion factor between units"""
    return converter.convert(1.0, from_unit, to_unit, commodity)

def list_commodities():
    """List all available commodities"""
    return converter.available_commodities

def list_units():
    """List common units"""
    return converter.available_units

# Example usage
if __name__ == "__main__":
    print("Modern Commodity Converter Examples\n" + "="*50)
    
    # Simple conversions
    print("\n1. Simple unit conversions (no commodity needed):")
    print(f"100 bbl = {convert(100, 'bbl', 'L'):.0f} L")
    print(f"1000 L = {convert(1000, 'L', 'bbl'):.2f} bbl")
    
    # Commodity-specific conversions
    print("\n2. Mass-Volume conversions (needs commodity):")
    print(f"100 kt diesel = {convert(100, 'kt', 'bbl', 'diesel'):.0f} bbl")
    print(f"1000 bbl gasoline = {convert(1000, 'bbl', 'mt', 'gasoline'):.2f} mt")
    
    # Energy conversions (simplified for now - needs more work)
    print("\n3. Energy conversions:")
    print("(Energy conversions need additional implementation work)")
    
    # Series with daily rates
    print("\n4. Pandas Series with rate conversions:")
    dates = pd.date_range('2024-01', periods=3, freq='MS')
    series = pd.Series([100, 110, 105], index=dates)
    result = convert(series, 'kt/month', 'bbl/day', 'diesel')
    print(f"January: {result.iloc[0]:.0f} bbl/day")
    
    # Available commodities
    print(f"\n5. Available commodities: {', '.join(list_commodities())}")
    
    # Error handling
    print("\n6. Error handling:")
    try:
        convert(100, 'kt', 'bbl')  # Missing commodity
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("Key improvements over original:")
    print("• Type hints and dataclasses for clarity")
    print("• Automatic dimensional analysis")
    print("• Clean separation of concerns") 
    print("• Caching for performance")
    print("• Better error messages")
    print("• Extensible commodity definitions")
    print("• Modern Python patterns")
