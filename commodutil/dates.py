import pandas as pd
import re
import datetime 

curmon = datetime.datetime.now().month
curyear = datetime.datetime.now().year
curmonyear = datetime.datetime(curyear, curmon, 1)
curmonyear_str = '{}-{}'.format(curyear, curmon) # get pandas time filtering


"""
Given a dataframe find the years in the column headings. Return a dict of colname to year
eg { 'Q1 2016' : 2016, 'Q1 2017' : 2017
"""
def find_year(df, use_delta=False):
    res = {}
    for colname in df:
        colregex = re.findall('\d\d\d\d', str(colname))
        if len(colregex) == 1:
            colyear = int(colregex[0])

        res[colname] = colyear
        if use_delta:
            delta = colyear - dates.curyear
            res[colname] = delta

    return res
