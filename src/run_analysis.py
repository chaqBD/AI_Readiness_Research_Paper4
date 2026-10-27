"""End-to-end analysis pipeline. Run from the project root:
    python src/run_analysis.py
Regenerates every table in outputs/tables/ and every figure in
outputs/figures/ from data/processed/analytic_cross_section.csv.
"""
import json
import warnings

import numpy as np
import pandas as pd

from constructs import INDICATORS, LABELS, load_analytic_sample, composite_scores
import descriptives as desc
import reliability as rel
import structural_model as sm_mod
import moderation as mod
import mediation as med
import visualize as viz

np.random.seed(42)
warnings.filterwarnings("ignore")

TABLES = "outputs/tables"


def main():
    full, complete = load_analytic_sample()
    comp = composite_scores(complete)
    df = pd.concat([complete.reset_index(drop=True), comp.reset_index(drop=True)], axis=1)
    print(f"Analytic sample: N = {len(df)} countries (full rebuilt panel: {len(full)})")

    # 1. Descriptives -----------------------------------------------------
    descriptives_table = desc.composite_descriptives(comp)
    descriptives_table.to_csv(f"{TABLES}/descriptives.csv", index=False)

    skew_fix = desc.econ_skew_before_after(df)
    skew_fix.to_csv(f"{TABLES}/econ_skew_before_after.csv", index=False)

    viz.distributions(comp, LABELS)
    viz.econ_skew_fix(
        (df["GDP_per_capita_const2015"] - df["GDP_per_capita_const2015"].mean()) / df["GDP_per_capita_const2015"].std(ddof=1),
        (df["log_GDP_per_capita"] - df["log_GDP_per_capita"].mean()) / df["log_GDP_per_capita"].std(ddof=1),
    )
    print("Descriptives written.")

    # 2. Correlation structure ---------------------------------------------
    corr = comp.rename(columns=LABELS).corr()
    corr.to_csv(f"{TABLES}/construct_correlations.csv")
    viz.correlation_heatmap(corr)
    print("Correlation matrix written.")

    # 3. Reliability & validity -------------------------------------------
    alphas = rel.cronbach_alphas(df)
    alphas.to_csv(f"{TABLES}/cronbach_alpha.csv", index=False)

    cfa_model = rel.fit_cfa(df)
    loadings = rel.standardized_loadings(cfa_model)
    loadings.to_csv(f"{TABLES}/cfa_standardized_loadings.csv", index=False)

    cr_ave = rel.cr_ave(loadings)
    cr_ave.to_csv(f"{TABLES}/cr_ave.csv", index=False)

    fl = rel.fornell_larcker(comp, cr_ave)
    fl.to_csv(f"{TABLES}/fornell_larcker.csv")

    htmt = rel.htmt(df)
    htmt.to_csv(f"{TABLES}/htmt.csv")

    import semopy
    cfa_fit = semopy.calc_stats(cfa_model)
    cfa_fit["SRMR"] = rel.srmr(cfa_model, df)
    cfa_fit.T.to_csv(f"{TABLES}/cfa_fit_indices.csv")
    print("Reliability & validity (alpha, CFA loadings, CR/AVE, Fornell-Larcker, HTMT) written.")

    # 4. Full structural SEM (diagnostic exhibit) --------------------------
    struct_model = sm_mod.fit_structural_model(df)
    struct_paths = sm_mod.structural_paths(struct_model)
    struct_paths.to_csv(f"{TABLES}/structural_sem_paths_UNSTABLE.csv", index=False)
    struct_fit = semopy.calc_stats(struct_model)
    struct_fit.T.to_csv(f"{TABLES}/structural_sem_fit_indices.csv")
    print("Full one-step structural SEM fitted (reported as a diagnostic exhibit; see report for instability caveat).")

    # 5. Moderated regression (H1a, H1b, H1c) ------------------------------
    d, model_main, model_full, vif, delta_r2 = mod.fit_moderation(df, x="AI", w="INST", y="SC")
    with open(f"{TABLES}/moderation_main_effects.txt", "w", encoding="utf-8") as f:
        f.write(model_main.summary().as_text())
    with open(f"{TABLES}/moderation_full_model.txt", "w", encoding="utf-8") as f:
        f.write(model_full.summary().as_text())
    vif.to_csv(f"{TABLES}/moderation_vif.csv", index=False)

    ci = model_full.conf_int()
    mod_coefs = pd.DataFrame({
        "term": model_full.params.index, "estimate": model_full.params.values,
        "se": model_full.bse.values, "p": model_full.pvalues.values,
        "ci_low": ci[0].values, "ci_high": ci[1].values,
    })
    mod_coefs.to_csv(f"{TABLES}/moderation_coefficients.csv", index=False)

    ss = mod.simple_slopes(model_full, df["INST"].mean(), df["INST"].std())
    ss.to_csv(f"{TABLES}/simple_slopes.csv", index=False)

    jn_bounds = mod.johnson_neyman(model_full)
    with open(f"{TABLES}/johnson_neyman_bounds.json", "w", encoding="utf-8") as f:
        json.dump({
            "bounds_centered_INST": jn_bounds,
            "bounds_raw_INST": [b + df["INST"].mean() for b in jn_bounds] if jn_bounds else None,
            "delta_r2_interaction": float(delta_r2),
        }, f, indent=2)

    viz.simple_slopes_plot(d, model_full)
    viz.johnson_neyman_plot(d, model_full, jn_bounds)
    print(f"Moderated regression written. Delta R^2 (interaction) = {delta_r2:.4f}, JN bounds = {jn_bounds}")

    # 6. Mediation (H2a) ----------------------------------------------------
    med_result = med.bootstrap_indirect(df, x="AI", m="SC", y="ECON")
    with open(f"{TABLES}/mediation_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(med_result, f, indent=2)
    pd.DataFrame([med_result]).to_csv(f"{TABLES}/mediation_bootstrap.csv", index=False)
    print(f"Bootstrap mediation written. Indirect effect = {med_result['indirect']:.3f} "
          f"95% CI [{med_result['indirect_ci_lower']:.3f}, {med_result['indirect_ci_upper']:.3f}], "
          f"significant = {med_result['significant']}")

    # 7. Combined path-coefficient forest plot ------------------------------
    rows = [
        {"label": "AI Readiness -> SC Performance", "estimate": model_full.params["AI_c"],
         "ci_low": ci[0]["AI_c"], "ci_high": ci[1]["AI_c"]},
        {"label": "Institutional Quality -> SC Performance", "estimate": model_full.params["INST_c"],
         "ci_low": ci[0]["INST_c"], "ci_high": ci[1]["INST_c"]},
        {"label": "AI x Institutions (interaction) -> SC", "estimate": model_full.params["interaction"],
         "ci_low": ci[0]["interaction"], "ci_high": ci[1]["interaction"]},
        {"label": "AI Readiness -> Economic Performance (direct, c')", "estimate": med_result["c_prime_direct"],
         "ci_low": None, "ci_high": None},
        {"label": "SC Performance -> Economic Performance (b, controlling AI)", "estimate": med_result["b_path"],
         "ci_low": None, "ci_high": None},
    ]
    # Fill CIs for mediation rows using the auxiliary OLS directly for consistency.
    import statsmodels.api as sm_api
    full_med_model = sm_api.OLS(df["ECON"], sm_api.add_constant(df[["AI", "SC"]])).fit()
    med_ci = full_med_model.conf_int()
    rows[3]["ci_low"], rows[3]["ci_high"] = med_ci.loc["AI", 0], med_ci.loc["AI", 1]
    rows[4]["ci_low"], rows[4]["ci_high"] = med_ci.loc["SC", 0], med_ci.loc["SC", 1]
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(f"{TABLES}/combined_path_coefficients.csv", index=False)
    viz.coefficient_forest(coef_df)
    print("Combined path-coefficient forest plot written.")

    print("\nAll outputs written to outputs/tables/ and outputs/figures/.")


if __name__ == "__main__":
    main()
