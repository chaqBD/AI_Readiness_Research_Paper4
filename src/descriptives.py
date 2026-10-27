"""Descriptive statistics and distributional diagnostics for the four
composite constructs, including the before/after comparison for the
log-transform fix applied to the Economic Performance construct."""
import numpy as np
import pandas as pd
from scipy import stats

from constructs import LABELS


def describe(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    rows = []
    for key, series in cols.items():
        s = series.dropna()
        shapiro_stat, shapiro_p = stats.shapiro(s)
        rows.append({
            "construct": key,
            "n": len(s),
            "mean": s.mean(),
            "sd": s.std(ddof=1),
            "min": s.min(),
            "max": s.max(),
            "skew": stats.skew(s),
            "kurtosis": stats.kurtosis(s),
            "shapiro_W": shapiro_stat,
            "shapiro_p": shapiro_p,
        })
    return pd.DataFrame(rows)


def composite_descriptives(composite_df: pd.DataFrame) -> pd.DataFrame:
    cols = {LABELS[c]: composite_df[c] for c in composite_df.columns}
    return describe(composite_df, cols)


def econ_skew_before_after(df: pd.DataFrame) -> pd.DataFrame:
    raw = (df["GDP_per_capita_const2015"] - df["GDP_per_capita_const2015"].mean()) / df["GDP_per_capita_const2015"].std(ddof=1)
    logged = (df["log_GDP_per_capita"] - df["log_GDP_per_capita"].mean()) / df["log_GDP_per_capita"].std(ddof=1)
    return describe(df, {"GDP per capita (raw, z-scored)": raw, "log(GDP per capita), z-scored": logged})
