import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from commodutil import dates
from commodutil import transforms


def curve_seasonal_zscore(hist, fwd):
    """
    Given some history for a timeseries and a forward curve, calculate the monthly
    z-score (std dev away from mean) along the forward curve
    """

    d = transforms.monthly_mean(hist).T.describe()

    if isinstance(fwd, pd.Series):
        fwd = pd.DataFrame(fwd)
    fwd["zscore"] = fwd.apply(
        lambda x: (d[x.name.month].loc["mean"] - x.iloc[0])
        / d[x.name.month].loc["std"],
        1,
    )
    return fwd


def reindex_zscore(df, range=10, calc_year_start: int = None):
    """
    Given a dataframe of contracts (or spreads), calculate z-score for current year onwards
    Essentially returns how far away the 'curve' is from historical trading range
    """
    df = df
    df = df.rename(
        columns={x: int(re.findall("\d\d\d\d", str(x))[0]) for x in df.columns}
    )  # turn columns into years
    d = df.loc[
        :, dates.curyear - range - 1 : dates.curyear - 1
    ]  # get subset of range years
    d = d[:-10]  # exclude last 10 rows to due to volatility close to expire

    dfs = []
    if not calc_year_start:
        calc_year_start = dates.curyear
    for year in df.loc[:, calc_year_start : df.columns[-1]]:
        z = (d.mean(axis=1) - df.loc[:, year]) / d.std(axis=1)
        z.name = year
        dfs.append(z)
    if len(dfs) > 0:
        res = pd.concat(dfs, axis=1)
        return res


@dataclass(frozen=True)
class PointStats:
    """
    Summary statistics for a single "as-of" point against a historical reference set.

    Percentile is returned as an empirical CDF in [0, 1], i.e. fraction of reference
    values <= current value.
    """

    asof: pd.Timestamp
    current_year: int | None
    current_value: float | None
    reference_years: list[int]
    reference_values: list[float]
    mean: float | None
    std: float | None
    zscore: float | None
    percentile: float | None


def last_value_at_or_before(
    series: pd.Series, asof: datetime | str | pd.Timestamp
) -> float | None:
    """
    Return the last non-null value at or before `asof`.

    Returns None if no value exists in the window.
    """
    ts = pd.Timestamp(asof)
    s = series.loc[:ts].dropna()
    if s.empty:
        return None
    val = s.iloc[-1]
    try:
        return float(val)
    except Exception:
        return None


def empirical_percentile(
    value: float, reference_values: Iterable[float]
) -> float | None:
    """
    Empirical percentile as fraction of reference values <= value.

    Returns None if reference is empty after filtering.
    """
    ref = [float(x) for x in reference_values if x is not None and not np.isnan(x)]
    if not ref:
        return None
    return float(np.mean([x <= value for x in ref]))


def point_stats(
    value: float | None, reference_values: Iterable[float]
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Compute (mean, std, zscore, percentile) for a value vs reference values.

    Std is sample std (ddof=1) when at least 2 reference values exist, else 0.0.
    """
    ref = np.array(
        [float(x) for x in reference_values if x is not None and not np.isnan(x)],
        dtype=float,
    )
    if value is None or np.isnan(value) or ref.size == 0:
        return None, None, None, None

    mean = float(np.mean(ref))
    std = float(np.std(ref, ddof=1)) if ref.size >= 2 else 0.0
    z = None if std == 0.0 else float((float(value) - mean) / std)
    p = empirical_percentile(float(value), ref.tolist())
    return mean, std, z, p


def select_reindex_prompt_column(df_reindexed: pd.DataFrame, *, within_days: int = 10):
    """
    Determine the "prompt" column for a reindex-year dataframe.

    Mirrors the legacy behavior in `commodplot.commodplotutil.reindex_year_df_rel_col`:
    - Prefer a column whose label contains `dates.curyear`.
    - If that column ends within `within_days` of the max x-date, prefer next year's column.
    """
    if df_reindexed is None or df_reindexed.empty:
        return None

    res_col = df_reindexed.columns[0]

    year_map = dates.find_year(df_reindexed)
    last_val_date = df_reindexed.index[-1]

    current_year_cols = [
        c for c in df_reindexed.columns if str(dates.curyear) in str(c)
    ]
    if not current_year_cols:
        return res_col

    res_col = current_year_cols[0]
    res_year = year_map.get(res_col)
    if not isinstance(res_year, int):
        return res_col

    relyear = pd.to_datetime(f"{res_year}-01-01")

    dft = df_reindexed[current_year_cols].dropna()
    if len(dft) == 0:
        return res_col

    relcol_series = df_reindexed[res_col].dropna()
    if relcol_series.empty:
        return res_col

    relcol_date = relcol_series.index[-1]
    delta = last_val_date - relcol_date
    if delta.days < within_days:
        relyear1 = (relyear + pd.DateOffset(years=1)).year
        relyear1_cols = [c for c in df_reindexed.columns if str(relyear1) in str(c)]
        if relyear1_cols:
            return relyear1_cols[0]

    return res_col


def reindex_year_point_stats(
    df: pd.DataFrame,
    *,
    asof: datetime | str | pd.Timestamp | None = None,
    lookback_years: int = 5,
    within_days: int = 10,
    trim_expiry: bool = False,
) -> PointStats:
    """
    Compute point stats for a reindex-year view.

    - Reindex to current year (`commodutil.transforms.reindex_year`).
    - Pick the prompt column via `select_reindex_prompt_column`.
    - Compare prompt value at `asof` vs prior `lookback_years` years at the same as-of date.

    If ``trim_expiry=True``, apply ``trim_expiry_noise`` to the reindexed frame
    before reading values, so that near-expiry volatile observations in reference
    years are excluded from the comparison.
    """
    if df is None or df.empty:
        return PointStats(
            asof=pd.NaT,
            current_year=None,
            current_value=None,
            reference_years=[],
            reference_values=[],
            mean=None,
            std=None,
            zscore=None,
            percentile=None,
        )

    dft = transforms.reindex_year(df)

    if trim_expiry:
        dft = trim_expiry_noise(dft)

    if dft is None or dft.empty:
        return PointStats(
            asof=pd.NaT,
            current_year=None,
            current_value=None,
            reference_years=[],
            reference_values=[],
            mean=None,
            std=None,
            zscore=None,
            percentile=None,
        )

    asof_ts = pd.Timestamp(dft.index.max()) if asof is None else pd.Timestamp(asof)

    prompt_col = select_reindex_prompt_column(dft, within_days=within_days)
    year_map = dates.find_year(dft)
    prompt_year = year_map.get(prompt_col) if prompt_col is not None else None
    prompt_year_int = prompt_year if isinstance(prompt_year, int) else None

    current_value = (
        last_value_at_or_before(dft[prompt_col], asof_ts)
        if prompt_col is not None
        else None
    )

    reference_years: list[int] = []
    reference_values: list[float] = []
    if prompt_year_int is not None:
        start_year = prompt_year_int - lookback_years
        end_year = prompt_year_int - 1
        for col in dft.columns:
            col_year = year_map.get(col)
            if not isinstance(col_year, int):
                continue
            if start_year <= col_year <= end_year:
                val = last_value_at_or_before(dft[col], asof_ts)
                if val is not None:
                    reference_years.append(col_year)
                    reference_values.append(val)

    mean, std, z, p = point_stats(current_value, reference_values)
    return PointStats(
        asof=asof_ts,
        current_year=prompt_year_int,
        current_value=current_value,
        reference_years=reference_years,
        reference_values=reference_values,
        mean=mean,
        std=std,
        zscore=z,
        percentile=p,
    )


def seasonal_point_stats(
    seas: pd.DataFrame,
    *,
    asof: datetime | str | pd.Timestamp | None = None,
    lookback_years: int = 5,
) -> PointStats:
    """
    Compute point stats from a seasonalized dataframe.

    Expects:
    - Index: current-year dates
    - Columns: years (ints)

    Uses `dates.curyear` as the current year column if present; otherwise uses max year column.
    """
    if seas is None or seas.empty:
        return PointStats(
            asof=pd.NaT,
            current_year=None,
            current_value=None,
            reference_years=[],
            reference_values=[],
            mean=None,
            std=None,
            zscore=None,
            percentile=None,
        )

    year_cols = [c for c in seas.columns if isinstance(c, (int, np.integer))]
    if not year_cols:
        return PointStats(
            asof=pd.NaT,
            current_year=None,
            current_value=None,
            reference_years=[],
            reference_values=[],
            mean=None,
            std=None,
            zscore=None,
            percentile=None,
        )

    current_year = dates.curyear if dates.curyear in year_cols else int(max(year_cols))
    asof_ts = pd.Timestamp(seas.index.max()) if asof is None else pd.Timestamp(asof)
    asof_ts = min(asof_ts, pd.Timestamp(seas.index.max()))

    # Use the last index <= asof, to avoid KeyError for non-trading days.
    valid_idx = seas.index[seas.index <= asof_ts]
    if len(valid_idx) == 0:
        return PointStats(
            asof=pd.NaT,
            current_year=int(current_year),
            current_value=None,
            reference_years=[],
            reference_values=[],
            mean=None,
            std=None,
            zscore=None,
            percentile=None,
        )

    asof_row = pd.Timestamp(valid_idx.max())
    try:
        current_value = float(seas.loc[asof_row, current_year])
    except Exception:
        current_value = None

    reference_years = [
        int(y)
        for y in year_cols
        if (current_year - lookback_years) <= int(y) <= (current_year - 1)
    ]
    reference_values: list[float] = []
    for y in reference_years:
        try:
            reference_values.append(float(seas.loc[asof_row, y]))
        except Exception:
            continue

    mean, std, z, p = point_stats(current_value, reference_values)
    return PointStats(
        asof=asof_row,
        current_year=int(current_year),
        current_value=current_value,
        reference_years=reference_years,
        reference_values=reference_values,
        mean=mean,
        std=std,
        zscore=z,
        percentile=p,
    )


def _base_label_from_column(col) -> str | None:
    """
    Derive a stable group key from a column label by stripping year-like tokens.

    Intended for grouping structures like:
    - "JunAug 2026" -> "JunAug"
    - "Q1Q2 2026" -> "Q1Q2"
    - "CAL 2025-2026" -> "CAL"

    Returns None if the label cannot be converted to a non-empty key.
    """
    s = str(col).strip()
    if not s:
        return None

    # Remove year ranges and single years (prefer full year tokens).
    s = re.sub(r"\b(19|20)\d{2}\s*-\s*(19|20)\d{2}\b", "", s)
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)

    # Cleanup leftover separators/spaces.
    s = re.sub(r"[-/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def reindex_year_point_stats_table(
    df: pd.DataFrame,
    *,
    asof: datetime | str | pd.Timestamp | None = None,
    lookback_years: int = 5,
    within_days: int = 10,
    min_columns: int = 3,
    trim_expiry: bool = False,
) -> pd.DataFrame:
    """
    Compute prompt-vs-history point stats for many structures in one dataframe.

    This is designed for frames where columns encode both a structure key and a year,
    e.g. "JunAug 2025", "JunAug 2026", "DecJan 2025", etc.

    The function:
    - groups columns by a "base label" (column label with year tokens stripped),
    - runs `reindex_year_point_stats` per group,
    - returns a sortable table (z-score/percentile) for scanning cheap/rich structures.

    If ``trim_expiry=True``, each per-group call applies expiry noise trimming.

    Notes:
    - Columns must include a 4-digit year somewhere for `dates.find_year` to work reliably.
    - Groups with fewer than `min_columns` columns are skipped.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "asof",
                "prompt_year",
                "value",
                "mean",
                "std",
                "zscore",
                "percentile",
                "n_reference",
            ]
        )

    groups: dict[str, list] = {}
    for col in df.columns:
        key = _base_label_from_column(col)
        if key is None:
            continue
        groups.setdefault(key, []).append(col)

    rows: list[dict] = []
    for key, cols in groups.items():
        if len(cols) < min_columns:
            continue
        stats_res = reindex_year_point_stats(
            df[cols],
            asof=asof,
            lookback_years=lookback_years,
            within_days=within_days,
            trim_expiry=trim_expiry,
        )
        rows.append(
            {
                "group": key,
                "asof": stats_res.asof,
                "prompt_year": stats_res.current_year,
                "value": stats_res.current_value,
                "mean": stats_res.mean,
                "std": stats_res.std,
                "zscore": stats_res.zscore,
                "percentile": stats_res.percentile,
                "n_reference": len(stats_res.reference_values),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "group",
                "asof",
                "prompt_year",
                "value",
                "mean",
                "std",
                "zscore",
                "percentile",
                "n_reference",
            ]
        )

    res = (
        pd.DataFrame(rows)
        .set_index("group")
        .sort_values(["zscore", "percentile"], ascending=[True, True])
    )
    return res


def prompt_strip_point_stats(
    df: pd.DataFrame,
    *,
    asof: datetime | str | pd.Timestamp | None = None,
    lookback_bdays: int = 756,
    require_all_columns: bool = True,
) -> pd.DataFrame:
    """
    Compute point stats per prompt-tenor column for a "prompt strip" dataframe.

    Intended shape:
      - index: DatetimeIndex (trading days)
      - columns: tenors (e.g., M1..M12) or any set of prompt labels

    The function:
      - selects an as-of date (default: latest date with values for all columns if `require_all_columns`)
      - for each column, computes (value, mean, std, zscore, percentile) vs a trailing window
        of `lookback_bdays` observations, excluding the current as-of value.

    Returns:
      A DataFrame indexed by column name with stats columns.
      The selected as-of timestamp is stored in `result.attrs["asof"]`.
    """
    if df is None or df.empty:
        res = pd.DataFrame(
            columns=["value", "mean", "std", "zscore", "percentile", "n_ref"]
        )
        res.attrs["asof"] = None
        return res

    dft = df.copy()
    dft.index = pd.to_datetime(dft.index)

    if asof is None:
        d = dft.dropna(how="any" if require_all_columns else "all")
        asof_ts = pd.Timestamp(d.index.max()) if not d.empty else None
    else:
        asof_ts = pd.Timestamp(asof)

    if asof_ts is None or pd.isna(asof_ts):
        res = pd.DataFrame(
            columns=["value", "mean", "std", "zscore", "percentile", "n_ref"]
        )
        res.attrs["asof"] = None
        return res

    rows: list[dict] = []
    for col in dft.columns:
        s = pd.to_numeric(dft[col], errors="coerce")
        s = s.dropna()
        current = last_value_at_or_before(s, asof_ts)
        if current is None:
            rows.append(
                {
                    "tenor": col,
                    "value": None,
                    "mean": None,
                    "std": None,
                    "zscore": None,
                    "percentile": None,
                    "n_ref": 0,
                }
            )
            continue

        hist = s.loc[:asof_ts].dropna()
        if lookback_bdays and len(hist) > 1:
            # Use exactly `lookback_bdays` observations prior to the current as-of value when possible.
            start = max(0, len(hist) - (lookback_bdays + 1))
            ref = hist.iloc[start:-1]
        else:
            ref = hist.iloc[:-1]

        mean, std, z, p = point_stats(float(current), ref.tolist())
        rows.append(
            {
                "tenor": col,
                "value": float(current),
                "mean": mean,
                "std": std,
                "zscore": z,
                "percentile": p,
                "n_ref": int(ref.shape[0]),
            }
        )

    res = pd.DataFrame(rows).set_index("tenor")
    res.attrs["asof"] = asof_ts
    return res


# ---------------------------------------------------------------------------
# Expiry noise detection and trimming
# ---------------------------------------------------------------------------


def detect_expiry_noise_cutoff(
    s: pd.Series,
    *,
    threshold_std: float = 2.0,
    min_stable_frac: float = 0.6,
    calm_streak: int = 3,
    min_obs: int = 20,
) -> pd.Timestamp | None:
    """
    Detect where expiry noise begins in a spread/fly time series.

    Spreads and flies become extremely volatile near expiry as liquidity dries
    up and individual legs settle independently.  This function finds the last
    "stable" date before the noisy tail by:

    1. Computing absolute daily changes.
    2. Establishing "normal" volatility from the first ``min_stable_frac``
       fraction of the data.
    3. Walking backwards from the end and looking for a contiguous run of days
       where the absolute change exceeds ``mean + threshold_std * std`` of the
       stable period.
    4. The cutoff is the last calm date before that noisy tail (requiring
       ``calm_streak`` consecutive calm days to confirm stability).

    Returns the cutoff date (last usable date), or None if no significant
    expiry noise is detected.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < min_obs:
        return None

    changes = s.diff().abs().dropna()
    if len(changes) < min_obs:
        return None

    # Establish normal volatility from first portion of data.
    stable_n = max(10, int(len(changes) * min_stable_frac))
    stable_changes = changes.iloc[:stable_n]
    mean_c = float(stable_changes.mean())
    std_c = float(stable_changes.std())

    if std_c == 0.0 or np.isnan(std_c):
        return None

    threshold = mean_c + threshold_std * std_c

    # Walk backwards: find where a sustained noisy tail begins.
    noisy = (changes > threshold).values
    n = len(noisy)
    calm_count = 0

    for i in range(n - 1, -1, -1):
        if not noisy[i]:
            calm_count += 1
            if calm_count >= calm_streak:
                # We found `calm_streak` consecutive calm days.
                # The cutoff is the last of these calm days (before noise resumes).
                cutoff_idx = i + calm_streak - 1
                if cutoff_idx >= n - 1:
                    return None  # No noise after this calm streak
                return pd.Timestamp(changes.index[cutoff_idx])
        else:
            calm_count = 0

    # Entire series is noisy (unlikely) — don't truncate.
    return None


def trim_expiry_noise(
    df: pd.DataFrame,
    *,
    threshold_std: float = 2.0,
    min_stable_frac: float = 0.6,
    calm_streak: int = 3,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Trim expiry noise from each column of a reindexed (or raw) DataFrame.

    For each column, detects the expiry-noise cutoff via
    ``detect_expiry_noise_cutoff`` and sets values after that date to NaN.

    Returns a copy of the DataFrame with noisy tails removed.
    """
    out = df.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        cutoff = detect_expiry_noise_cutoff(
            s,
            threshold_std=threshold_std,
            min_stable_frac=min_stable_frac,
            calm_streak=calm_streak,
            min_obs=min_obs,
        )
        if cutoff is not None:
            out.loc[out.index > cutoff, col] = np.nan
    return out
