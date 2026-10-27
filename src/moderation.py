"""Moderated regression for H1a/H1b/H1c: does Institutional Quality moderate
the AI Readiness -> SC Performance relationship?

Upgrades over the original SPSS pilot's plain-OLS hierarchical regression:
heteroskedasticity-robust (HC3) standard errors, VIF multicollinearity
diagnostics, simple-slopes at +/-1SD of the moderator, and a full
Johnson-Neyman interval identifying exactly where the AI->SC slope is
significant, rather than a single interaction p-value.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def fit_moderation(df: pd.DataFrame, x: str = "AI", w: str = "INST", y: str = "SC"):
    d = df.copy()
    d[f"{x}_c"] = d[x] - d[x].mean()
    d[f"{w}_c"] = d[w] - d[w].mean()
    d["interaction"] = d[f"{x}_c"] * d[f"{w}_c"]

    predictors = [f"{x}_c", f"{w}_c", "interaction"]
    X = sm.add_constant(d[predictors])
    model_main = sm.OLS(d[y], sm.add_constant(d[predictors[:2]])).fit(cov_type="HC3")
    model_full = sm.OLS(d[y], X).fit(cov_type="HC3")

    vif = pd.DataFrame({
        "variable": predictors,
        "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(predictors))],
    })

    delta_r2 = model_full.rsquared - model_main.rsquared
    return d, model_main, model_full, vif, delta_r2


def simple_slopes(model_full, w_mean: float, w_sd: float, x: str = "AI_c", w: str = "INST_c"):
    b_x = model_full.params[x]
    b_inter = model_full.params["interaction"]
    cov = model_full.cov_params()
    var_bx, var_inter = cov.loc[x, x], cov.loc["interaction", "interaction"]
    cov_bx_inter = cov.loc[x, "interaction"]
    df_resid = model_full.df_resid

    rows = []
    for label, w_val in [("-1 SD", -w_sd), ("Mean", 0.0), ("+1 SD", w_sd)]:
        slope = b_x + w_val * b_inter
        se = np.sqrt(var_bx + 2 * w_val * cov_bx_inter + w_val ** 2 * var_inter)
        t = slope / se
        p = 2 * (1 - stats.t.cdf(abs(t), df_resid))
        rows.append({"moderator_level": label, "INST_value": w_mean + w_val, "slope": slope, "se": se, "t": t, "p": p})
    return pd.DataFrame(rows)


def johnson_neyman(model_full, x: str = "AI_c", alpha: float = 0.05):
    b_x = model_full.params[x]
    b_inter = model_full.params["interaction"]
    cov = model_full.cov_params()
    var_bx, var_inter = cov.loc[x, x], cov.loc["interaction", "interaction"]
    cov_bx_inter = cov.loc[x, "interaction"]
    df_resid = model_full.df_resid
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)

    a = b_inter ** 2 - (t_crit ** 2) * var_inter
    b = 2 * b_x * b_inter - (t_crit ** 2) * 2 * cov_bx_inter
    c = b_x ** 2 - (t_crit ** 2) * var_bx

    disc = b ** 2 - 4 * a * c
    if disc < 0 or a == 0:
        return None
    w1 = (-b + np.sqrt(disc)) / (2 * a)
    w2 = (-b - np.sqrt(disc)) / (2 * a)
    return tuple(sorted((w1, w2)))
