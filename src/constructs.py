"""Shared construct definitions, indicator-level prep, and composite scoring."""
import numpy as np
import pandas as pd

DATA_PATH = "data/processed/analytic_cross_section.csv"

INDICATORS = {
    "AI": ["Avg_AI_Readiness", "Internet_users_pct_pop"],
    "SC": [
        "LPI_timeliness", "LPI_infrastructure", "LPI_tracking_tracing",
        "LPI_customs", "LPI_intl_shipments", "LPI_logistics_quality",
    ],
    "ECON": ["log_GDP_per_capita", "Exports_pct_GDP", "Trade_openness"],
    "INST": [
        "WGI_govt_effectiveness", "WGI_rule_of_law",
        "WGI_regulatory_quality", "WGI_control_corruption",
    ],
}

LABELS = {
    "AI": "ξ1 AI Readiness",
    "SC": "η1 SC Performance",
    "ECON": "η2 Economic Performance",
    "INST": "ξ2 Institutional Quality",
}

ALL_INDICATORS = [c for cols in INDICATORS.values() for c in cols]
RAW_ECON_COLS = ["GDP_per_capita_const2015", "Exports_pct_GDP", "Trade_openness"]


def load_analytic_sample():
    """Loads the rebuilt cross-section, applies the log-GDP fix, and returns
    (full_df, complete_df) where complete_df is listwise-complete on every
    indicator used by the four constructs."""
    df = pd.read_csv(DATA_PATH)
    df["log_GDP_per_capita"] = np.log(df["GDP_per_capita_const2015"])
    complete = df.dropna(subset=ALL_INDICATORS).reset_index(drop=True)
    return df, complete


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Naive composite z-score indices (mean of standardised items), kept only
    as the baseline comparator against the original SPSS pilot's method -- the
    CFA factor scores (structural_model.py) are used for all hypothesis tests."""
    out = pd.DataFrame(index=df.index)
    for construct, cols in INDICATORS.items():
        z = df[cols].apply(zscore)
        out[construct] = z.mean(axis=1)
    return out
