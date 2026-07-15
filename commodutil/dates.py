import datetime
import re
from datetime import datetime, date, time, timedelta

from dateutil.relativedelta import relativedelta


def _curmon():
    return datetime.now().month


def _curyear():
    return datetime.now().year


def _curmonyear():
    return datetime(_curyear(), _curmon(), 1)


def _curmonyear_str():
    return "%s-%s" % (_curyear(), _curmon())  # get pandas time filtering


def _last_day_of_prev_month():
    return date.today().replace(day=1) - relativedelta(days=1)


def _start_day_of_prev_month():
    return date.today().replace(day=1) - relativedelta(months=1)


def _prevmon():
    return _start_day_of_prev_month().month


def _prevmon_str():
    sdpm = _start_day_of_prev_month()
    return "%s-%s" % (sdpm.year, sdpm.month)  # get pandas time filtering


def _nextyear():
    return _curyear() + 1


def _prevyear():
    return _curyear() - 1


# Compute "current date" values live on each attribute access (PEP 562) so that a
# long-running process (Prefect worker, Dash app) that imports this module once still
# sees the correct month/year after a boundary, instead of a stale import-time snapshot.
_DYNAMIC = {
    "curmon": _curmon,
    "curyear": _curyear,
    "curmonyear": _curmonyear,
    "curmonyear_str": _curmonyear_str,
    "last_day_of_prev_month": _last_day_of_prev_month,
    "start_day_of_prev_month": _start_day_of_prev_month,
    "prevmon": _prevmon,
    "prevmon_str": _prevmon_str,
    "nextyear": _nextyear,
    "prevyear": _prevyear,
}


def __getattr__(name):
    try:
        return _DYNAMIC[name]()
    except KeyError:
        raise AttributeError(name)


def find_year(df, use_delta=False):
    """
    Given a dataframe find the years in the column headings. Return a dict of colname to year
    eg { 'Q1 2016' : 2016, 'Q1 2017' : 2017
    """
    res = {}
    for colname in df:
        colregex = re.findall(r"\d\d\d\d", str(colname))
        colyear = None
        if len(colregex) >= 1:
            colyear = int(colregex[0])

        if colyear:
            res[colname] = colyear
            if colyear and use_delta:
                delta = colyear - _curyear()
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
