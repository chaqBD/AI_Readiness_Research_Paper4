"""Measurement model: Cronbach's alpha, CFA loadings, CR/AVE, discriminant validity."""
import numpy as np
import pandas as pd
import pingouin as pg
import semopy

from constructs import INDICATORS, LABELS

MEASUREMENT_MODEL = """
AI =~ Avg_AI_Readiness + Internet_users_pct_pop
SC =~ LPI_timeliness + LPI_infrastructure + LPI_tracking_tracing + LPI_customs + LPI_intl_shipments + LPI_logistics_quality
ECON =~ log_GDP_per_capita + Exports_pct_GDP + Trade_openness
INST =~ WGI_govt_effectiveness + WGI_rule_of_law + WGI_regulatory_quality + WGI_control_corruption
"""


def fit_cfa(df: pd.DataFrame):
    model = semopy.Model(MEASUREMENT_MODEL)
    model.fit(df)
    return model


def cronbach_alphas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for construct, cols in INDICATORS.items():
        alpha, ci = pg.cronbach_alpha(data=df[cols])
        rows.append({
            "construct": LABELS[construct], "items": len(cols),
            "alpha": alpha, "ci_low": ci[0], "ci_high": ci[1],
        })
    return pd.DataFrame(rows)


def standardized_loadings(model: semopy.Model) -> pd.DataFrame:
    ins = model.inspect(std_est=True)
    loadings = ins[(ins["op"] == "~") & (ins["rval"].isin(INDICATORS.keys()))].copy()
    loadings = loadings.rename(columns={"rval": "construct", "lval": "indicator", "Est. Std": "loading"})
    loadings["loading"] = loadings["loading"].astype(float)
    return loadings[["construct", "indicator", "loading", "p-value"]]


def cr_ave(loadings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for construct in INDICATORS:
        lam = loadings.loc[loadings["construct"] == construct, "loading"].values
        sum_lam = lam.sum()
        sum_lam_sq = (lam ** 2).sum()
        sum_err = (1 - lam ** 2).sum()
        cr = sum_lam ** 2 / (sum_lam ** 2 + sum_err)
        ave = sum_lam_sq / len(lam)
        rows.append({"construct": LABELS[construct], "CR": cr, "AVE": ave, "sqrt_AVE": np.sqrt(ave)})
    return pd.DataFrame(rows)


def fornell_larcker(composite_df: pd.DataFrame, ave: pd.DataFrame) -> pd.DataFrame:
    """composite_df must have columns named by the short construct codes
    (AI, SC, ECON, INST), as produced by constructs.composite_scores()."""
    sqrt_ave = ave.set_index("construct")["sqrt_AVE"]
    codes = list(INDICATORS.keys())
    corr = composite_df[codes].corr()
    fl = corr.rename(index=LABELS, columns=LABELS).copy()
    for code in codes:
        fl.loc[LABELS[code], LABELS[code]] = sqrt_ave[LABELS[code]]
    return fl


def htmt(df: pd.DataFrame) -> pd.DataFrame:
    """Heterotrait-Monotrait ratio of correlations (Henseler et al., 2015)."""
    constructs = list(INDICATORS.keys())
    result = pd.DataFrame(index=[LABELS[c] for c in constructs], columns=[LABELS[c] for c in constructs], dtype=float)
    item_corr = df[[c for cols in INDICATORS.values() for c in cols]].corr().abs()

    def mean_hetero(c1, c2):
        pairs = [item_corr.loc[i, j] for i in INDICATORS[c1] for j in INDICATORS[c2]]
        return np.mean(pairs)

    def mean_mono(c):
        cols = INDICATORS[c]
        if len(cols) < 2:
            return np.nan
        vals = [item_corr.loc[i, j] for idx, i in enumerate(cols) for j in cols[idx + 1:]]
        return np.mean(vals)

    for i, c1 in enumerate(constructs):
        for c2 in constructs:
            if c1 == c2:
                result.loc[LABELS[c1], LABELS[c2]] = np.nan
                continue
            denom = np.sqrt(mean_mono(c1) * mean_mono(c2))
            result.loc[LABELS[c1], LABELS[c2]] = mean_hetero(c1, c2) / denom if denom and not np.isnan(denom) else np.nan
    return result


def srmr(model: semopy.Model, df: pd.DataFrame) -> float:
    obs_vars = model.vars["observed"]
    sigma_model = model.calc_sigma()[0]
    sigma_model = pd.DataFrame(sigma_model, index=obs_vars, columns=obs_vars)
    sample_cov = df[obs_vars].cov()

    def to_corr(cov):
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)

    r_model = to_corr(sigma_model.values)
    r_sample = to_corr(sample_cov.values)
    n = len(obs_vars)
    iu = np.triu_indices(n, k=1)
    resid = r_sample[iu] - r_model[iu]
    return float(np.sqrt(np.mean(resid ** 2)))
