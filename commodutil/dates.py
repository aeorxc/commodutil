import datetime
import re
import time
from datetime import datetime, date, timedelta

curmon = datetime.now().month
curyear = datetime.now().year
curmonyear = datetime(curyear, curmon, 1)
curmonyear_str = "%s-%s" % (curyear, curmon)  # get pandas time filtering

last_day_of_prev_month = date.today().replace(day=1) - timedelta(days=1)
start_day_of_prev_month = date.today().replace(day=1) - timedelta(
    days=last_day_of_prev_month.day
)

prevmon = start_day_of_prev_month.month
prevmon_str = "%s-%s" % (
    start_day_of_prev_month.year,
    start_day_of_prev_month.month,
)  # get pandas time filtering

nextyear = curyear + 1
prevyear = curyear - 1


def find_year(df, use_delta=False):
    """
    Given a dataframe find the years in the column headings. Return a dict of colname to year
    eg { 'Q1 2016' : 2016, 'Q1 2017' : 2017
    """
    res = {}
    for colname in df:
        colregex = re.findall("\d\d\d\d", str(colname))
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


def time_until_end_of_day(dt=None):
    # type: (datetime.datetime) -> datetime.timedelta
    """
    Get timedelta until end of day on the datetime passed, or current time.
    """
    if dt is None:
        dt = datetime.now()
    tomorrow = dt + timedelta(days=1)
    return (datetime.combine(tomorrow, time.min) - dt).seconds
