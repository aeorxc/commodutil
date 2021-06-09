import re
import datetime 

curmon = datetime.datetime.now().month
curyear = datetime.datetime.now().year
curmonyear = datetime.datetime(curyear, curmon, 1)
curmonyear_str = '%s-%s' % (curyear, curmon) # get pandas time filtering

nextyear = curyear + 1


def find_year(df, use_delta=False):
    """
    Given a dataframe find the years in the column headings. Return a dict of colname to year
    eg { 'Q1 2016' : 2016, 'Q1 2017' : 2017
    """
    res = {}
    for colname in df:
        colregex = re.findall('\d\d\d\d', str(colname))
        colyear = None
        if len(colregex) >= 1:
            colyear = int(colregex[0])

        if colyear:
            res[colname] = colyear
            if colyear and use_delta:
                delta = colyear - curyear
                res[colname] = delta
        else:
            res[colname] = colname

    return res
