####
# Commodity conversion factors
#
# - Convention:
#       commodity_from_to
#
# eg: diesel_kt_bbl
#
# optionally specify parameters for non standard entities: eg diesel_kt_bbl_iea
#
# standard should be to multiply the conversion factor
# eg 1 mt * 7.45 = 7.45 bbls
#
# bbl = barrels
# km3 = cubmic meters
# kt = kilo atonne
# ltr = litre
# gj = 1000 mj # thus km3_gj is same as l_mj


import pandas as pd


# converts different names for commodities to standard used in this file
commonnamemap = {
    'dies': 'diesel',
    'ulsd': 'diesel',
    'go': 'diesel',
    'gas': 'gasoline', # tricky - could be NatGag
    'etho': 'ethanol',

}

# API
api_modulus = 141.5

# bbls to km3
km3_bbl = 6.28981

# Gallons to litres
# 1 US gallon = 0.83267384 Imp gallon = 3.78541178 Liters
gal_ltr = 3.78541178


# Light Ends
naphtha_kt_bbl = 8.9

gasoline_density_kg_l = 0.745
gasoline_kt_bbl = 8.33
gasoline_kt_km3 = 1/gasoline_density_kg_l  # 1.342281879
gasoline_km3_bbl = gasoline_kt_bbl / gasoline_kt_km3
gasoline_gal_kt = gal_ltr * (gasoline_density_kg_l / 1000)
gasoline_km3_gj = 32


ccs_kt_bbl = 8.62
isomerate_kt_bbl = 9.53
alkylate_kt_bbl = 8.99
reformate_kt_bbl = 7.07


ethanol_density_kg_l = 0.789
ethanol_kt_bbl = 8.33  #7.96
ethanol_kt_km3 = 1/ethanol_density_kg_l # 1.2674
ethanol_km3_bbl = ethanol_kt_bbl / ethanol_kt_km3
ethanol_gal_kt = gal_ltr * (ethanol_density_kg_l / 1000)
ethanol_km3_gj = 21


etbe_density_kg_l = 0.745
etbe_kt_km3 = 1/ethanol_density_kg_l # 1.34
etbe_km3_gj = 27.04

# Middle distillates
jet_kt_bbl = 7.88

diesel_density_kg_l = 0.843
diesel_kt_bbl = 7.45
diesel_kt_km3 = 1/diesel_density_kg_l  # 1.18623962
diesel_km3_bbl = diesel_kt_bbl / diesel_kt_km3
diesel_gal_kt = gal_ltr * (diesel_density_kg_l / 1000)
diesel_km3_gj = 36


fame_density_kg_l = 0.892
fame_kt_bbl = 7.051345
fame_kt_km3 = 1/ fame_density_kg_l  # 1.121076233
fame_km3_bbl = fame_kt_bbl / diesel_kt_km3
fame_gal_kt = gal_ltr * (fame_density_kg_l / 1000)
fame_km3_gj = 33


b7_density_kg_l = 0.845
b7_kt_bbl = 7.078
b7_kt_km3 = 1/ b7_density_kg_l  # 1.11834
b7_km3_bbl = b7_kt_bbl / diesel_kt_km3
b7_gal_kt = gal_ltr * (b7_density_kg_l / 1000)
b7_km3_gj = 35.8


hvo_density_kg_l = 0.778
hvo_kt_bbl = 8.046
hvo_kt_km3 = 1/hvo_density_kg_l  # 1.285347044
hvo_km3_bbl = hvo_kt_bbl / hvo_kt_km3
hvo_gal_kt = gal_ltr * (hvo_density_kg_l / 1000)
hvo_km3_gj = 34


# FO / Heave
vgo_kt_bbl = 6.9
sr_fo_kt_bbl = 6.65
fo_kt_bbl = 6.35


# Crude


# Natgas

natgas_density_kg_l = 0.542
natgas_kt_bbl = 11.6
natgas_kt_km3 = 1/natgas_density_kg_l  # 1.844
natgas_km3_bbl = natgas_kt_bbl / natgas_kt_km3
natgas_gal_kt = gal_ltr * (natgas_density_kg_l / 1000)
natgas_km3_gj = 26.137 #https://callmepower.ca/en/faq/gigajoule-cubic-metre-gas


# print diesel_km3_bbl
# print fame_gal_mt
# print gasoline_gal_kt
# print diesel_gal_kt

def _stdcmmd(cmmdstring):
    global commonnamemap
    cmmdstring = cmmdstring.lower()
    cmmdstring = commonnamemap.get(cmmdstring, cmmdstring)
    return cmmdstring


def _stdunits(unit):
    if unit == 'mt':
        return 'kt'
    if unit == 'm3':
        return 'km3'

    return unit


def convfactor(commodity, fromunit, tounit):
    commodity = _stdcmmd(commodity)

    # TODO - move to mt/m3 convetion rather than kt/km3
    fromunit = _stdunits(fromunit)
    tounit = _stdunits(tounit)

    factor = '{}_{}_{}'.format(commodity, fromunit, tounit)
    if factor in globals():
        return globals()[factor]

    factor = '{}_{}_{}'.format(commodity, tounit, fromunit)
    if factor in globals():
        res = globals()[factor]
        res = 1/res
        return res

    # if af this point we don't have a conversion factor try to mix an match on kt/km3 combinations
    if fromunit == 'kt':
        int_cv = convfactor(commodity, 'km3', tounit)
        kt_km3_cv = convfactor(commodity, 'kt', 'km3')
        new_cv = int_cv * kt_km3_cv
        return new_cv

    if tounit == 'kt':
        int_cv = convfactor(commodity, 'km3', fromunit)
        kt_km3_cv = convfactor(commodity, 'kt', 'km3')
        new_cv = 1/(int_cv * kt_km3_cv)
        return new_cv

    raise ValueError('no conversion factor for {} from: {} to: {}'.format(commodity, fromunit, tounit))


def convert_days_in_month(s, transform='mul'):
    m = pd.Series(s.index.days_in_month, index=s.index)
    if transform == 'mul':
        s = s.mul(m, 0)
    elif transform == 'div':
        s = s.div(m, 0)
    return s


def convert(val, commodity, fromunit=None, tounit=None):
    if fromunit != tounit:
        assert commodity is not None, 'need type of commodity to convert {} from {} to {}'.format(val, fromunit, tounit)
        fromunit_i, tounit_i = fromunit, tounit
        if fromunit.endswith('/d'):
            fromunit_i = fromunit_i.replace('/d', '')
        if tounit.endswith('/d'):
            tounit_i = tounit_i.replace('/d', '')
        unitconvfactor = convfactor(commodity, fromunit_i, tounit_i)

        if fromunit.endswith('/d'):
            val = convert_days_in_month(val, 'mul')

        val = val * unitconvfactor

        if tounit.endswith('/d'):
            val = convert_days_in_month(val, 'div')

    return val


if __name__ == '__main__':
    a = convfactor('diesel', 'kt', 'km3')
    print(a)