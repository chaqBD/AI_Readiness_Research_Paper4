# AI Readiness and Supply Chain Performance: A Cross-National Pilot Study

**Author:** Shakir Adeyemi Tewogbade

A cross-national pilot study examining whether government AI readiness drives supply chain (logistics) performance, whether institutional quality moderates that relationship, and whether supply chain performance mediates the link between AI readiness and economic performance, using Structural Equation Modelling concepts applied across a sample of over 140 countries. The empirical analysis is implemented entirely in Python (no SPSS/AMOS), covering data reconciliation, confirmatory factor analysis, reliability/validity testing, robust moderated regression, and bootstrap mediation.

## Repository structure

```
Dataset/                     Raw source data (World Bank WDI/WGI/LPI, Oxford AI Readiness, ITU, Stanford HAI)
data/processed/              Rebuilt analytic cross-section + data reconciliation notes
src/                         Python analysis pipeline
  build_dataset.py             Rebuilds the analytic cross-section from the raw panel
  constructs.py                Construct/indicator definitions and composite scoring
  descriptives.py              Descriptive statistics and normality diagnostics
  reliability.py                Cronbach's alpha, CFA loadings, CR/AVE, Fornell-Larcker, HTMT
  structural_model.py          Full CFA + structural-paths SEM (semopy)
  moderation.py                 HC3-robust moderated regression, VIF, simple slopes, Johnson-Neyman
  mediation.py                  Bootstrap mediation analysis
  visualize.py                  All figures
  run_analysis.py              End-to-end pipeline entry point
outputs/tables/               All numeric results (CSV/JSON/text)
outputs/figures/              All figures (PNG)
reports/results_report.md     Full write-up of methods and results
```

## Reproducing the analysis

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python src/build_dataset.py   # rebuilds data/processed/analytic_cross_section.csv
python src/run_analysis.py    # regenerates every table in outputs/tables/ and figure in outputs/figures/
```

Results are deterministic (`numpy` seed 42 fixed wherever bootstrapping is used).

## Key findings

- Analytic sample: **N = 141 countries**, rebuilt from the raw World Bank panel with a documented, auditable "most recent available year per indicator" reconciliation rule (see `data/processed/DATA_NOTES.md`).
- **H1a supported**: AI Readiness significantly predicts SC Performance (β = 0.53, p < .001).
- **H1b supported**: Institutional Quality significantly predicts SC Performance (β = 0.37, p < .001).
- **H1c supported, with the opposite sign to the earlier SPSS-based pilot**: Institutional Quality **strengthens** (not weakens) the AI Readiness → SC Performance relationship (interaction β = 0.22, p < .001), robust to plain OLS and to removing the most influential countries.
- **H2a not supported**: SC Performance does not mediate AI Readiness → Economic Performance (bootstrap 95% CI for the indirect effect spans zero), matching the original pilot's conclusion but via a modern bootstrap test rather than the Sobel test.

Full results, figures, and discussion are in [`reports/results_report.md`](reports/results_report.md).

## Methodological improvements over the original SPSS pilot

1. Rebuilt and audited the data pipeline (the original pre-built dataset had zero usable Logistics Performance Index observations).
2. Fixed the Economic Performance construct's severe skew/kurtosis via a log-transform of GDP per capita.
3. Removed a circular indicator (`LPI_overall`, a composite of the other six LPI items used in the same construct).
4. Estimated reliability/validity from a fitted confirmatory factor model (standardised loadings, CR, AVE) rather than simple composite correlations.
5. Added HTMT discriminant validity testing (Henseler et al., 2015), which surfaces a construct-overlap issue the original Fornell-Larcker check missed.
6. Diagnosed and reported the instability of a one-step latent structural model at this sample size, motivating a defensible two-step (CFA-then-regression) strategy.
7. Used heteroskedasticity-robust (HC3) standard errors and VIF multicollinearity diagnostics in place of uncorrected OLS.
8. Replaced the Sobel test with bias-corrected bootstrap confidence intervals for the mediation analysis.
9. Added Johnson-Neyman region-of-significance analysis for the moderation effect.

## Data sources

World Bank World Development Indicators (WDI), Worldwide Governance Indicators (WGI), and Logistics Performance Index (LPI); Oxford Insights Government AI Readiness Index; International Telecommunication Union (ITU) digital skills indicators; Stanford HAI AI Index.
