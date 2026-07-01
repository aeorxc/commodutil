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

    # Select prompt BEFORE trimming so rollover logic sees complete data
    prompt_col = select_reindex_prompt_column(dft, within_days=within_days)

    if trim_expiry:
        exclude = [prompt_col] if prompt_col is not None else None
        dft = trim_expiry_noise(dft, exclude_columns=exclude)

    asof_ts = pd.Timestamp(dft.index.max()) if asof is None else pd.Timestamp(asof)
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


# ---------------------------------------------------------------------------
# Expired-structure detection
# ---------------------------------------------------------------------------

_QUARTER_LAST_MONTH = {"q1": 3, "q2": 6, "q3": 9, "q4": 12}


def _front_delivery_month(base_label):
    """Extract front delivery month number from label like 'JanFeb', 'Q1Q2'.

    Uses month_abbr_inv from commodutil.forward.util for month parsing
    (same pattern as spread_combination_fly in forwards.py).
    """
    from commodutil.forward.util import month_abbr_inv

    s = base_label.strip().lower()
    if not s or len(s) < 2:
        return None
    # Monthly: starts with 3-letter month abbreviation
    if len(s) >= 3 and s[:3] in month_abbr_inv:
        return month_abbr_inv[s[:3]]
    # Quarterly: starts with Q + digit
    m = re.match(r"q(\d)", s)
    if m:
        return {"1": 1, "2": 4, "3": 7, "4": 10}.get(m.group(1))
    return None  # CAL, H1H2, SummerWinter — don't filter


def _is_structure_expired(base_label, year, ref_date=None):
    """Check if a structure's front delivery leg has expired.

    Monthly structures: expired when front month has passed.
    Quarterly structures: expired when entire front quarter has passed.
    CAL/H1/H2/Summer/Winter: never filtered (partial expiry is normal).
    """
    if ref_date is None:
        ref_date = datetime.now()
    if isinstance(ref_date, pd.Timestamp):
        ref_date = ref_date.to_pydatetime()
    front_month = _front_delivery_month(base_label)
    if front_month is None:
        return False
    s = base_label.strip().lower()
    # Quarterly: expired after the last month of the front quarter
    q_match = re.match(r"q(\d)", s)
    if q_match:
        last_m = _QUARTER_LAST_MONTH.get(f"q{q_match.group(1)}", front_month)
        boundary = (
            datetime(year + 1, 1, 1) if last_m == 12 else datetime(year, last_m + 1, 1)
        )
    else:
        # Monthly: expired after the front month ends
        boundary = (
            datetime(year + 1, 1, 1)
            if front_month == 12
            else datetime(year, front_month + 1, 1)
        )
    return ref_date >= boundary


def reindex_year_point_stats_table(
    df: pd.DataFrame,
    *,
    asof: datetime | str | pd.Timestamp | None = None,
    lookback_years: int = 5,
    within_days: int = 10,
    min_columns: int = 3,
    trim_expiry: bool = False,
    skip_expired: bool = True,
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
    If ``skip_expired=True`` (default), structures whose front delivery month has
    already expired are excluded from results.

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

    effective_asof = pd.Timestamp(asof) if asof is not None else None
    if effective_asof is None:
        dft_for_asof = transforms.reindex_year(df)
        if dft_for_asof is not None and not dft_for_asof.empty:
            effective_asof = pd.Timestamp(dft_for_asof.index.max())

    ref_date = (
        effective_asof.to_pydatetime()
        if effective_asof is not None and not pd.isna(effective_asof)
        else datetime.now()
    )

    rows: list[dict] = []
    for key, cols in groups.items():
        if len(cols) < min_columns:
            continue
        if skip_expired and _is_structure_expired(
            key, dates.curyear, ref_date=ref_date
        ):
            continue
        stats_res = reindex_year_point_stats(
            df[cols],
            asof=effective_asof,
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
    seasonal_window_days: int | None = None,
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

    Parameters:
      seasonal_window_days: If set, filter reference values to ±N calendar days around the
        same day-of-year as the as-of date. This converts the z-score from unconditional
        ("is $9.30 cheap vs ALL M1 history?") to seasonal ("is $9.30 cheap for mid-February?").
        CRITICAL for seasonal products like gasoline cracks, gasnaph, or any structure with
        strong intra-year patterns. Typical value: 21 (±3 weeks = ~6 week window).

    Returns:
      A DataFrame indexed by column name with stats columns.
      The selected as-of timestamp is stored in `result.attrs["asof"]`.
      If seasonal_window_days is set, `result.attrs["seasonal_window_days"]` is also stored.
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

        # Seasonal filtering: keep only reference values within ±N calendar days
        # of the same day-of-year as the as-of date.  This prevents unconditional
        # z-scores from confusing seasonal patterns with cheap/rich signals.
        if seasonal_window_days is not None and not ref.empty:
            asof_doy = asof_ts.dayofyear
            ref_doy = ref.index.dayofyear
            # Circular distance (handles year-wrap, e.g., Jan vs Dec)
            dist = (ref_doy - asof_doy).to_series(index=ref.index).abs()
            dist = dist.where(dist <= 182, 365 - dist)
            ref = ref[dist.values <= seasonal_window_days]

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
    if seasonal_window_days is not None:
        res.attrs["seasonal_window_days"] = seasonal_window_days
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


# ---------------------------------------------------------------------------
# STL-detrended seasonal z-scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class STLZScoreResult:
    """
    Result of STL-detrended seasonal z-score analysis.

    The STL (Seasonal-Trend decomposition using LOESS) approach separates a
    time series into Trend + Seasonal + Residual, then computes z-scores on
    the residuals grouped by their within-year period.  This gives a
    detrended, seasonally-adjusted measure of surprise.

    Background & references:
        - Cleveland et al. (1990), "STL: A Seasonal-Trend Decomposition
          Procedure Based on Loess", Journal of Official Statistics, 6(1).
        - risktools-dev (Python port of R's RTL package) implements a similar
          ``chart_zscore`` using STL + per-period residual z-scores.
          See: https://github.com/bbcho/risktools-dev
        - statsmodels ``STL`` class:
          https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html

    Why this matters for commodity fundamentals:
        Standard seasonal overlays (5-year range charts) conflate structural
        trends with seasonal patterns.  For example, if crude stocks drew down
        steadily over 5 years, the current level looks "low vs history" even
        if it's *above* the trend line.  STL decomposition removes the trend
        first, so the z-score reflects genuine deviation from seasonal norms.

    When to use vs existing stats functions:
        - ``reindex_year_point_stats`` / ``seasonal_point_stats``: best for
          spreads, flies, and structures that don't have strong multi-year
          trends.
        - ``stl_seasonal_zscore``: best for level data with trends — stocks,
          demand, refinery runs, production, any fundamentals indicator.
    """

    date: pd.Timestamp
    value: float | None
    trend: float | None
    seasonal: float | None
    residual: float | None
    zscore: float | None
    percentile: float | None
    period_mean: float | None
    period_std: float | None
    period_label: int | None
    n_reference: int


def stl_seasonal_zscore(
    s: pd.Series,
    *,
    freq: str | None = None,
    period: int | None = None,
    seasonal: int | None = None,
    robust: bool = True,
    asof: datetime | str | pd.Timestamp | None = None,
    min_obs: int = 52,
) -> STLZScoreResult:
    """
    Compute a detrended, seasonally-adjusted z-score using STL decomposition.

    Decomposes the series into Trend + Seasonal + Residual via STL (LOESS),
    then computes the z-score of the current residual relative to all
    historical residuals at the same within-year period (e.g., same ISO week
    or same month).

    This is directly inspired by risktools-dev ``chart_zscore`` (which ports
    R's RTL package), adapted to fit the commodutil conventions (returns a
    structured result rather than a chart, uses the same sign convention as
    ``point_stats``: positive z = above mean = richer/higher than normal).

    Parameters
    ----------
    s : pd.Series
        Time series with a DatetimeIndex.  Should be regularly spaced
        (weekly or monthly works best).  Daily data is supported but will
        be slower and may need a larger ``seasonal`` window.
    freq : str, optional
        Resample frequency before decomposition.  Common values:
        ``'W-FRI'`` (weekly, Friday), ``'MS'`` (month start), ``'ME'``
        (month end).  If None, uses the series as-is (must have a
        recognisable frequency).
    period : int, optional
        Seasonal period for STL.  Auto-detected if not provided:
        52 for weekly, 12 for monthly, 365 for daily.
    seasonal : int, optional
        Length of the seasonal smoother (must be odd, >= 3).
        Auto-set if not provided: 13 for monthly, 53 for weekly,
        7 for daily.  Larger = smoother seasonal component.
    robust : bool, default True
        Use robust fitting (downweights outliers in the LOESS fit).
        Generally recommended for commodity data which has occasional
        spikes.
    asof : datetime-like, optional
        Evaluation date.  Defaults to the last date in the series.
    min_obs : int, default 52
        Minimum observations required.  STL needs at least 2 full
        seasonal cycles to produce meaningful results.

    Returns
    -------
    STLZScoreResult
        Frozen dataclass with the decomposition values at ``asof``,
        the z-score, and the empirical percentile vs same-period
        historical residuals.

    Examples
    --------
    >>> from pyoilfundydb.fundamentals import FundamentalHandler
    >>> fh = FundamentalHandler()
    >>> stocks = fh.series(indexids=[12345], start_date=date(2018, 1, 1))
    >>> stocks_weekly = stocks.iloc[:, 0]  # single series
    >>> result = stl_seasonal_zscore(stocks_weekly, freq='W-FRI')
    >>> print(f"z={result.zscore:.2f}, pct={result.percentile:.0%}")

    >>> # For EIA weekly crude stocks:
    >>> result = stl_seasonal_zscore(eia_crude_stocks, freq='W-FRI')
    >>> if result.zscore and result.zscore > 2.0:
    ...     print("Stocks significantly above detrended seasonal norm (bearish)")

    Notes
    -----
    - The z-score sign convention matches ``point_stats``:
      positive = above the period mean (for stocks: bearish; for demand: bullish).
    - ``robust=True`` (default) differs from risktools which uses ``robust=False``.
      Robust fitting is preferred for commodity data where occasional supply
      disruptions or weather events create outliers that shouldn't dominate
      the decomposition.
    - The period label is the within-year period number: ISO week (1-53) for
      weekly data, month (1-12) for monthly data, day-of-year for daily.
    """
    try:
        from statsmodels.tsa.seasonal import STL as _STL
    except ImportError:
        raise ImportError(
            "statsmodels is required for stl_seasonal_zscore. "
            "Install it with: pip install statsmodels"
        )

    if s is None or len(s) < min_obs:
        return STLZScoreResult(
            date=pd.NaT,
            value=None,
            trend=None,
            seasonal=None,
            residual=None,
            zscore=None,
            percentile=None,
            period_mean=None,
            period_std=None,
            period_label=None,
            n_reference=0,
        )

    # Ensure Series with DatetimeIndex
    ts = s.copy()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()

    if freq is not None:
        ts = ts.resample(freq).last().dropna()

    if len(ts) < min_obs:
        return STLZScoreResult(
            date=pd.NaT,
            value=None,
            trend=None,
            seasonal=None,
            residual=None,
            zscore=None,
            percentile=None,
            period_mean=None,
            period_std=None,
            period_label=None,
            n_reference=0,
        )

    # Auto-detect period and seasonal smoother length
    inferred = pd.infer_freq(ts.index)
    if period is None:
        if inferred and inferred.startswith("W"):
            period = 52
        elif inferred and inferred.startswith("M"):
            period = 12
        elif inferred and inferred.startswith("D"):
            period = 365
        else:
            # Guess from median gap between observations
            median_days = ts.index.to_series().diff().median().days
            if median_days is not None:
                if median_days <= 2:
                    period = 365
                elif median_days <= 10:
                    period = 52
                else:
                    period = 12
            else:
                period = 52  # safe default for commodity data

    if seasonal is None:
        if period >= 200:
            seasonal = 7  # daily
        elif period >= 30:
            seasonal = 53  # weekly
        else:
            seasonal = 13  # monthly

    # Run STL decomposition
    stl = _STL(
        ts,
        period=period,
        seasonal=seasonal,
        seasonal_deg=0,
        robust=robust,
    )
    result = stl.fit()

    resid = result.resid
    trend = result.trend
    seasonal_component = result.seasonal

    # Assign within-year period labels for grouping residuals
    if period >= 200:
        period_labels = resid.index.dayofyear
    elif period >= 30:
        period_labels = resid.index.isocalendar().week.astype(int).values
    else:
        period_labels = resid.index.month

    resid_df = pd.DataFrame(
        {
            "residual": resid,
            "period": period_labels,
        }
    )

    # Determine as-of date
    asof_ts = pd.Timestamp(ts.index.max()) if asof is None else pd.Timestamp(asof)
    # Find closest available date
    valid_idx = resid_df.index[resid_df.index <= asof_ts]
    if len(valid_idx) == 0:
        return STLZScoreResult(
            date=asof_ts,
            value=None,
            trend=None,
            seasonal=None,
            residual=None,
            zscore=None,
            percentile=None,
            period_mean=None,
            period_std=None,
            period_label=None,
            n_reference=0,
        )

    asof_actual = valid_idx.max()
    current_resid = float(resid_df.loc[asof_actual, "residual"])
    current_period = int(resid_df.loc[asof_actual, "period"])

    # Get all residuals at the same period, excluding current observation
    same_period = resid_df[
        (resid_df["period"] == current_period) & (resid_df.index < asof_actual)
    ]["residual"].dropna()

    ref_values = same_period.tolist()
    mean, std, z, p = point_stats(current_resid, ref_values)

    return STLZScoreResult(
        date=asof_actual,
        value=float(ts.loc[asof_actual]) if asof_actual in ts.index else None,
        trend=float(trend.loc[asof_actual]) if asof_actual in trend.index else None,
        seasonal=float(seasonal_component.loc[asof_actual])
        if asof_actual in seasonal_component.index
        else None,
        residual=current_resid,
        zscore=z,
        percentile=p,
        period_mean=mean,
        period_std=std,
        period_label=current_period,
        n_reference=len(ref_values),
    )


def stl_seasonal_zscore_series(
    s: pd.Series,
    *,
    freq: str | None = None,
    period: int | None = None,
    seasonal: int | None = None,
    robust: bool = True,
    min_obs: int = 52,
) -> pd.DataFrame:
    """
    Compute STL-detrended z-scores for every observation in a series.

    Same methodology as ``stl_seasonal_zscore`` but applied to every point,
    returning a full DataFrame suitable for charting (e.g., a z-score bar
    chart or heatmap).

    Each observation's z-score is computed against all *prior* observations
    at the same within-year period (expanding window, no lookahead).

    Parameters
    ----------
    s : pd.Series
        Time series with DatetimeIndex.
    freq, period, seasonal, robust, min_obs :
        Same as ``stl_seasonal_zscore``.

    Returns
    -------
    pd.DataFrame
        Columns: value, trend, seasonal, residual, period, zscore, percentile.
        Index: DatetimeIndex matching the (resampled) input.
    """
    try:
        from statsmodels.tsa.seasonal import STL as _STL
    except ImportError:
        raise ImportError(
            "statsmodels is required for stl_seasonal_zscore_series. "
            "Install it with: pip install statsmodels"
        )

    if s is None or len(s) < (min_obs if min_obs else 52):
        return pd.DataFrame(
            columns=[
                "value",
                "trend",
                "seasonal",
                "residual",
                "period",
                "zscore",
                "percentile",
            ]
        )

    ts = s.copy()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()

    if freq is not None:
        ts = ts.resample(freq).last().dropna()

    if len(ts) < (min_obs if min_obs else 52):
        return pd.DataFrame(
            columns=[
                "value",
                "trend",
                "seasonal",
                "residual",
                "period",
                "zscore",
                "percentile",
            ]
        )

    # Auto-detect period (same logic as stl_seasonal_zscore)
    inferred = pd.infer_freq(ts.index)
    if period is None:
        if inferred and inferred.startswith("W"):
            period = 52
        elif inferred and inferred.startswith("M"):
            period = 12
        elif inferred and inferred.startswith("D"):
            period = 365
        else:
            median_days = ts.index.to_series().diff().median().days
            if median_days is not None:
                if median_days <= 2:
                    period = 365
                elif median_days <= 10:
                    period = 52
                else:
                    period = 12
            else:
                period = 52

    if seasonal is None:
        if period >= 200:
            seasonal = 7
        elif period >= 30:
            seasonal = 53
        else:
            seasonal = 13

    stl = _STL(ts, period=period, seasonal=seasonal, seasonal_deg=0, robust=robust)
    result = stl.fit()

    if period >= 200:
        period_labels = result.resid.index.dayofyear
    elif period >= 30:
        period_labels = result.resid.index.isocalendar().week.astype(int).values
    else:
        period_labels = result.resid.index.month

    out = pd.DataFrame(
        {
            "value": ts,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "period": period_labels,
        }
    )

    # Compute expanding-window z-scores (no lookahead)
    zscores = []
    percentiles = []
    for i in range(len(out)):
        row_period = out.iloc[i]["period"]
        current_resid = out.iloc[i]["residual"]
        prior = out.iloc[:i]
        ref = prior[prior["period"] == row_period]["residual"].dropna().tolist()
        _, _, z, p = point_stats(current_resid, ref)
        zscores.append(z)
        percentiles.append(p)

    out["zscore"] = zscores
    out["percentile"] = percentiles
    return out


def trim_expiry_noise(
    df: pd.DataFrame,
    *,
    exclude_columns: Iterable | None = None,
    threshold_std: float = 2.0,
    min_stable_frac: float = 0.6,
    calm_streak: int = 3,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Trim expiry noise from each column of a reindexed (or raw) DataFrame.

    For each column, detects the expiry-noise cutoff via
    ``detect_expiry_noise_cutoff`` and sets values after that date to NaN.

    Columns listed in ``exclude_columns`` are left untouched (useful for
    preserving the prompt/current year column).

    Returns a copy of the DataFrame with noisy tails removed.
    """
    out = df.copy()
    skip = set(exclude_columns) if exclude_columns else set()
    for col in out.columns:
        if col in skip:
            continue
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
