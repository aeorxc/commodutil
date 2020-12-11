from commodutil import transforms
from commodutil import dates
import re
import pandas as pd


def curve_seasonal_zscore(hist, fwd):
    """
    Given some history for a timeseries and a forward curve, calculate the monthly
    z-score (std dev away from mean) along the forward curve
    """

    d = transforms.monthly_mean(hist).T.describe()

    if isinstance(fwd, pd.Series):
        fwd = pd.DataFrame(fwd)
    fwd['zscore'] = fwd.apply(lambda x: (d[x.name.month].loc['mean'] - x.iloc[0]) / d[x.name.month].loc['std'], 1)
    return fwd


def reindex_zscore(df, range=10):
    """
    Given a dataframe of contracts (or spreads), calculate z-score for current year onwards
    Essentially returns how far away the 'curve' is from historical trading range
    """
    df = transforms.reindex_year(df)
    df = df.rename(columns={x: int(re.findall('\d\d\d\d', str(x))[0]) for x in df.columns})  # turn columns into years
    d = df.loc[:, dates.curyear - range - 1:dates.curyear - 1]  # get subset of range years
    d = d[:-10] # exclude last 10 rows to due to volatility close to expire

    dfs = []
    for year in df.loc[:, dates.curyear:df.columns[-1]]:
        z = (d.mean(axis=1) - df.loc[:, year]) / d.std(axis=1)
        z.name = year
        dfs.append(z)
    res = pd.concat(dfs, 1)

    return res