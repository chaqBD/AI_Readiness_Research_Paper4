"""Mediation test for H2a (SC performance mediating AI readiness -> Economic
performance), using bias-corrected bootstrap CIs for the indirect effect
instead of the Sobel test used in the original SPSS pilot.

The Sobel test assumes the sampling distribution of the product term a*b is
normal, which is frequently violated (the product of two normals is itself
not normal) and gives the test low statistical power; bias-corrected
bootstrapping is the modern recommended standard (Preacher & Hayes, 2004;
Hayes, 2018) and makes no distributional assumption about a*b.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

N_BOOT = 5000
SEED = 42


def _paths(df: pd.DataFrame, x: str, m: str, y: str):
    a_model = sm.OLS(df[m], sm.add_constant(df[x])).fit()
    a = a_model.params[x]

    full_model = sm.OLS(df[y], sm.add_constant(df[[x, m]])).fit()
    b = full_model.params[m]
    c_prime = full_model.params[x]

    total_model = sm.OLS(df[y], sm.add_constant(df[x])).fit()
    c_total = total_model.params[x]

    return a, b, c_prime, c_total, a_model, full_model, total_model


def bootstrap_indirect(df: pd.DataFrame, x: str, m: str, y: str, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(df)
    a_obs, b_obs, c_prime_obs, c_total_obs, a_model, full_model, total_model = _paths(df, x, m, y)
    indirect_obs = a_obs * b_obs

    boot_indirect = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        a_b, b_b, *_ = _paths(sample, x, m, y)
        boot_indirect[i] = a_b * b_b

    # Bias-corrected percentile CI
    z0 = stats.norm.ppf((boot_indirect < indirect_obs).mean())
    alpha = 0.05
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(2 * z0 + z_lo)
    p_hi = stats.norm.cdf(2 * z0 + z_hi)
    ci_lo, ci_hi = np.percentile(boot_indirect, [100 * p_lo, 100 * p_hi])

    return {
        "a_path": a_obs, "a_se": a_model.bse[x], "a_p": a_model.pvalues[x],
        "b_path": b_obs, "b_se": full_model.bse[m], "b_p": full_model.pvalues[m],
        "c_prime_direct": c_prime_obs, "c_prime_se": full_model.bse[x], "c_prime_p": full_model.pvalues[x],
        "c_total": c_total_obs, "c_total_se": total_model.bse[x], "c_total_p": total_model.pvalues[x],
        "indirect": indirect_obs,
        "indirect_ci_lower": ci_lo, "indirect_ci_upper": ci_hi,
        "significant": not (ci_lo < 0 < ci_hi),
        "n_boot": n_boot,
    }
