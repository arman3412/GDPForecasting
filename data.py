import io
import os
import time
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


# Configuration

TARGET_RESOLUTION = "quarterly"          # this pipeline is quarterly-only for now
FREQ_STRINGS = {"quarterly": "QS"}       # quarter-start, matches OECD Q dating
PERIODS_PER_YEAR = {"quarterly": 4}


OECD_COUNTRIES = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT",
    "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
}


class IndicatorRequirements:
    """Shape we want a series in AFTER processing (mirrors the FRED version).

    pct_change : produce a percent-change column (annualized if requested)
    diff       : produce a first-difference column
    level      : keep the resampled raw level as a column
    annualized : when pct_change is True, annualize it
    """

    def __init__(self, pct_change=False, diff=False, level=True, annualized=False):
        self.pct_change = pct_change
        self.diff = diff
        self.level = level
        self.annualized = annualized


# Indicators

# Each entry: logical_name -> dict(url, freq, requirements)
# url: the FULL OECD Data Explorer "Developer API" URL

# req: native frequency of the series, 'M' (monthly) or 'Q' (quarterly).
# Monthly series are aggregated to quarters by averaging.
# requirements: how to transform the raw series into model features.

INDICATORS = {
    # Composite leading indicator (strongest single GDP predictor)
    # Coverage is G20 + Spain only (~20 countries) -- this caps the panel.
    "CLI": {
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI/.M.LI...AA...H?dimensionAtObservation=AllDimensions",
        "freq": "M",
        "requirements": IndicatorRequirements(diff=True, level=True),
    },

    # Harmonized unemployment rate (monthly)
    "UNEMP": {
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M/..._Z.Y._T.Y_GE15..M?dimensionAtObservation=AllDimensions",
        "freq": "M",
        "requirements": IndicatorRequirements(diff=True, level=True),
    },

    # Consumer price index, all items, NOT seasonally adjusted (monthly)
    "CPI": {
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL/.M.N.CPI.IX._T.._Z?dimensionAtObservation=AllDimensions",
        "freq": "M",
        "requirements": IndicatorRequirements(pct_change=True, annualized=True, level=False),
    },

    # Long-term (10y govt) interest rate, % per annum (monthly)
    # Verified working: keep the ",4.0" pin and the "_Z._Z._Z" activity code.
    "LTRATE": {
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/.M.IRLT.PA._Z._Z._Z?dimensionAtObservation=AllDimensions",
        "freq": "M",
        "requirements": IndicatorRequirements(diff=True, level=True),
    },

    # Industrial production volume index, NSA (monthly)
    "INDPRO": {
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_INDSERV,4.3/.M.PRVM.IX.BTE.N...?dimensionAtObservation=AllDimensions",
        "freq": "M",
        "requirements": IndicatorRequirements(pct_change=True, annualized=True, level=False),
    },
}

# Indicators in this set don't gate the join: a country missing one of these
# stays in the panel instead of being dropped. CLI is G20+Spain only (~20
# countries) and was capping the whole panel at 9 countries via the inner
# join -- see report_intersection output. Missing values for an optional
# indicator are filled with 0 plus a "<name>_missing" flag column (see
# assemble_panel), so the model can learn to discount it per-country rather
# than that country disappearing entirely.
OPTIONAL_INDICATORS = {"CLI"}

# Target: quarterly real GDP growth, explicit country list (avoids aggregate
# codes like OECD/G20/EA19 entirely -- belt-and-suspenders with OECD_COUNTRIES).
TARGET = {
    "url": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1/Q..ZAF+SAU+RUS+IDN+IND+CHN+BRA+ARG+AUS+AUT+BEL+CAN+CHE+CHL+COL+CRI+CZE+DEU+DNK+ESP+FIN+EST+FRA+GBR+GRC+HUN+ISL+LTU+IRL+ISR+ITA+JPN+KOR+LUX+LVA+MEX+NLD+NOR+NZL+POL+PRT+SVK+SVN+SWE+TUR+USA.S1..B1GQ......G1.?dimensionAtObservation=AllDimensions",
    "freq": "Q",
    "already_growth": True,
}


# ---------------------------------------------------------------------------
# Fetch layer (full Developer API URL -> tidy long DataFrame)
# ---------------------------------------------------------------------------

def _normalize_url(url):
    """Force a machine-readable, full-history CSV pull.

    Keeps the user's pasted query params but: strips any time limits the Data
    Explorer baked in (which would silently truncate history), and forces
    format=csvfile + dimensionAtObservation=AllDimensions.
    """
    parts = urlsplit(url.strip())
    params = dict(parse_qsl(parts.query))
    for k in ("startPeriod", "endPeriod", "lastNObservations", "firstNObservations"):
        params.pop(k, None)
    params["format"] = "csvfile"
    params["dimensionAtObservation"] = "AllDimensions"
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                       urlencode(params), parts.fragment))


def _parse_oecd_period(s):
    """Parse OECD TIME_PERIOD strings to quarter-start Timestamps.

    Handles 'YYYY-Qn' (quarterly), 'YYYY-MM' (monthly), and 'YYYY' (annual).
    Everything is snapped to the start of its quarter so series align.
    """
    s = str(s).strip()
    if "-Q" in s:
        year, q = s.split("-Q")
        return pd.Timestamp(int(year), (int(q) - 1) * 3 + 1, 1)
    if "-" in s:  # YYYY-MM
        year, month = s.split("-")[:2]
        return pd.Timestamp(int(year), int(month), 1)
    return pd.Timestamp(int(s), 1, 1)  # YYYY


def fetch_oecd(url, retries=3, pause=2.0):
    """Fetch one indicator for ALL countries from a full Developer API URL.

    Paste the URL exactly as the Data Explorer's 'Developer API' button gives it
    -- no splitting into ref/key. Returns a tidy long DataFrame [country, date,
    value]. 
    """
    full_url = _normalize_url(url)

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(full_url, timeout=60,
                                headers={"Accept": "application/vnd.sdmx.data+csv"})
            if resp.status_code == 404:
                raise RuntimeError(
                    f"404 -- the server rejected this URL:\n  {full_url}\n"
                )
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text))
            break
        except Exception as e:                       # noqa: BLE001
            last_err = e
            time.sleep(pause * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to fetch:\n  {full_url}\n{last_err}")

    # SDMX-CSV standard column names. Fall back loudly if the schema shifts.
    cols = {c.upper(): c for c in raw.columns}
    try:
        country_col = cols["REF_AREA"]
        time_col = cols["TIME_PERIOD"]
        value_col = cols["OBS_VALUE"]
    except KeyError:
        raise RuntimeError(
            f"Unexpected columns: {list(raw.columns)}. "
            f"Expected REF_AREA, TIME_PERIOD, OBS_VALUE. URL was:\n  {full_url}"
        )

    out = pd.DataFrame({
        "country": raw[country_col].astype(str),
        "date": raw[time_col].map(_parse_oecd_period),
        "value": pd.to_numeric(raw[value_col], errors="coerce"),
    })
    out = out[out["country"].isin(OECD_COUNTRIES)]  # drop aggregates (OECD, G20, EA19, ...)
    return out.dropna(subset=["value"]).sort_values(["country", "date"])


# Reshape & per-country transforms


def to_quarterly(long_df, native_freq):
    """Collapse a tidy long frame to quarterly resolution, PER COUNTRY.

    Monthly series are averaged within each quarter; quarterly series pass
    through. Resampling is done per country so quarter boundaries never mix
    two countries.
    """
    frames = []
    for country, grp in long_df.groupby("country"):
        s = grp.set_index("date")["value"].sort_index()
        s = s.resample("QS").mean()  # monthly -> quarter mean; quarterly -> align
        frames.append(pd.DataFrame({"country": country, "date": s.index, "value": s.values}))
    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def transform_per_country(panel, requirements, annualize_periods=4):
    """Apply level/diff/pct_change transforms WITHIN each country.

    Every operation is grouped by country so the first observation of one
    country is never differenced against the last observation of the previous
    one. Returns {suffix: tidy frame [country, date, <suffix value>]}.
    """
    panel = panel.sort_values(["country", "date"])
    g = panel.groupby("country")["value"]
    out = {}

    if requirements.level:
        out["level"] = panel[["country", "date", "value"]].rename(columns={"value": "level"})

    if requirements.diff:
        d = panel.copy()
        d["diff"] = g.diff().values
        out["diff"] = d[["country", "date", "diff"]]

    if requirements.pct_change:
        p = panel.copy()
        pct = g.pct_change()
        if requirements.annualized:
            pct = (1 + pct) ** annualize_periods - 1
        out["pct_change"] = (
            p.assign(pct_change=(pct.values * 100))[["country", "date", "pct_change"]]
        )

    return out


def build_feature_panel(indicators=INDICATORS, offline_frames=None):
    """Fetch -> quarterly -> per-country transform for every indicator.

    Returns a long frame [country, date, feature, value]. 
    offline_frames lets tests inject pre-built tidy frames instead of hitting the network:
    {logical_name: tidy_frame[country,date,value]}.
    """
    pieces = []
    for name, spec in indicators.items():
        print(f"  Fetching {name}...")
        if offline_frames is not None:
            raw_long = offline_frames[name]
        else:
            raw_long = fetch_oecd(spec["url"])
        quarterly = to_quarterly(raw_long, spec["freq"])
        for suffix, tidy in transform_per_country(quarterly, spec["requirements"]).items():
            val_col = tidy.columns[-1]
            tidy = tidy.rename(columns={val_col: "value"})
            tidy["feature"] = f"{name}_{suffix}"
            pieces.append(tidy[["country", "date", "feature", "value"]])
    return pd.concat(pieces, ignore_index=True)


def build_target_panel(target=TARGET, offline_frame=None):
    """Build the GDP-growth target, per country, as a tidy frame.

    If the target series is already a growth rate, pass it through. Otherwise
    treat it as a volume index and compute annualized quarter by quarter growth per country.
    """
    print("  Fetching GDP target...")
    raw_long = offline_frame if offline_frame is not None \
        else fetch_oecd(target["url"])
    quarterly = to_quarterly(raw_long, target["freq"])
    quarterly = quarterly.sort_values(["country", "date"])

    if target.get("already_growth"):
        quarterly = quarterly.rename(columns={"value": "GDP_target"})
    else:
        g = quarterly.groupby("country")["value"].pct_change()
        quarterly["GDP_target"] = ((1 + g) ** 4 - 1).values * 100
        quarterly = quarterly.drop(columns="value")
    return quarterly[["country", "date", "GDP_target"]]


# Assemble the (country, date) matrix


def assemble_panel(feature_long, target_long, min_quarters=40,
                   standardize_levels=False, optional_indicators=OPTIONAL_INDICATORS):
    """Pivot features wide, join the target, align on (country, date).

    min_quarters       : drop any country with fewer than this many complete rows
                         after the join (too short to window + split sensibly).
    standardize_levels : if True, z-score each *_level column within its country.
    optional_indicators : indicator names that should not gate the join. A country
                         missing one of these stays in the panel; its columns for
                         that indicator are filled with 0 and a "<name>_missing"
                         flag column (1.0 = missing) is added so the model can
                         learn to discount it rather than the country being
                         dropped entirely. Required indicators still cause the
                         row to be dropped via dropna if missing.
    """
    wide = feature_long.pivot_table(
        index=["country", "date"], columns="feature", values="value"
    ).reset_index()

    merged = wide.merge(target_long, on=["country", "date"], how="inner")
    merged = merged.sort_values(["country", "date"])

    optional_cols = [c for c in merged.columns
                     if any(c.startswith(name + "_") for name in optional_indicators)]
    required_cols = [c for c in merged.columns
                     if c not in optional_cols and c not in ("country", "date")]

    # Drop rows missing a REQUIRED column; optional columns are handled below.
    merged = merged.dropna(subset=required_cols)

    # Optional indicators: flag + zero-fill instead of dropping the country.
    for base in optional_indicators:
        cols = [c for c in optional_cols if c.startswith(base + "_")]
        if not cols:
            continue
        is_missing = merged[cols].isna().any(axis=1)
        merged[f"{base}_missing"] = is_missing.astype(float)
        merged[cols] = merged[cols].fillna(0.0)

    # Drop countries with too little history to be usable.
    counts = merged.groupby("country")["date"].transform("size")
    merged = merged[counts >= min_quarters]

    if standardize_levels:
        level_cols = [c for c in merged.columns if c.endswith("_level")]
        for c in level_cols:
            merged[c] = merged.groupby("country")[c].transform(
                lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
            )

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Windowing -- the part that silently corrupts the model if done wrong
# ---------------------------------------------------------------------------

def make_windows(panel, lookback=8, feature_cols=None, target_col="GDP_target"):
    """Build supervised (X, y) windows

    For each country, sort by date and slide a window of length `lookback` over
    the features; the target is the GDP growth of the quarter immediately after
    the window. Returns:
        X       : (n_samples, lookback, n_features) float array
        y       : (n_samples,) float array
        groups  : (n_samples,) array of country codes  -- use this to do a
                  GROUP-AWARE train/test split (never split a country's window
                  across train and test, and ideally hold out whole countries
                  and/or the most recent quarters).
        dates   : (n_samples,) array of the TARGET date for each sample
    """
    if feature_cols is None:
        feature_cols = [c for c in panel.columns
                        if c not in ("country", "date")]

    X, y, groups, dates = [], [], [], []
    for country, grp in panel.groupby("country"):
        grp = grp.sort_values("date")
        feats = grp[feature_cols].to_numpy(dtype=float)
        tgt = grp[target_col].to_numpy(dtype=float)
        d = grp["date"].to_numpy()
        for t in range(lookback, len(grp)):
            X.append(feats[t - lookback:t])
            y.append(tgt[t])
            groups.append(country)
            dates.append(d[t])

    if not X:
        return (np.empty((0, lookback, len(feature_cols))),
                np.empty(0), np.empty(0, dtype=object), np.empty(0, dtype="datetime64[ns]"))
    return (np.asarray(X), np.asarray(y),
            np.asarray(groups, dtype=object), np.asarray(dates))




def report_indicator_coverage(feature_long, target_long=None):
    # Per-indicator and target country coverage, before the inner join.
  
    print(f"{'Feature':<20} {'Countries':>10}")
    print("-" * 32)
    base_names = sorted(set(c.split("_")[0] for c in feature_long["feature"].unique()))
    for base in base_names:
        cols = [c for c in feature_long["feature"].unique() if c.startswith(base + "_")]
        countries = feature_long[feature_long["feature"].isin(cols)]["country"].unique()
        print(f"{base:<20} {len(countries):>10}")
    if target_long is not None:
        print(f"{'GDP_target':<20} {target_long['country'].nunique():>10}")


def report_intersection(feature_long, target_long):
    """Show exactly which countries survive all indicators + target, and which
    single indicator is responsible for dropping each country that doesn't.
    """
    base_names = sorted(set(c.split("_")[0] for c in feature_long["feature"].unique()))
    country_sets = {}
    for base in base_names:
        cols = [c for c in feature_long["feature"].unique() if c.startswith(base + "_")]
        country_sets[base] = set(feature_long[feature_long["feature"].isin(cols)]["country"].unique())
    country_sets["GDP_target"] = set(target_long["country"].unique())

    all_countries = sorted(set.union(*country_sets.values()))
    survivors = set.intersection(*country_sets.values())
    print(f"Countries present in ALL {len(country_sets)} sources: {len(survivors)}")
    print(f"  {sorted(survivors)}")
    print()
    print("Countries missing from at least one source (and which ones drop them):")
    for c in all_countries:
        if c in survivors:
            continue
        missing_from = [name for name, s in country_sets.items() if c not in s]
        if len(missing_from) <= 2:  # only show near-misses, not universally-missing
            print(f"  {c}: missing from {missing_from}")


def report_panel(panel):
    n_countries = panel["country"].nunique()
    per_country = panel.groupby("country")["date"].size().sort_values()
    print(f"Panel: {len(panel)} country-quarters, "
          f"{n_countries} countries, {panel.shape[1] - 2} feature columns")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Shortest country: {per_country.index[0]} ({per_country.iloc[0]} q)")
    print(f"Longest  country: {per_country.index[-1]} ({per_country.iloc[-1]} q)")
    print(f"Median quarters/country: {int(per_country.median())}")



if __name__ == "__main__":
    feature_long = build_feature_panel()
    target_long = build_target_panel()
    print()
    report_indicator_coverage(feature_long, target_long)
    print()
    report_intersection(feature_long, target_long)
    print()
    panel = assemble_panel(feature_long, target_long)
    report_panel(panel)

    X, y, groups, dates = make_windows(panel, lookback=8)
    print(f"\nWindows: X={X.shape}, y={y.shape}, "
          f"{len(set(groups))} countries represented")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "oecd_panel.csv")
    panel.to_csv(out_path, index=False)
    print(f"Saved panel to {out_path}")