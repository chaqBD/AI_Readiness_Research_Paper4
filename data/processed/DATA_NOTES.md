# Data reconciliation notes

## Source problem

`Dataset/CONSOLIDATED_MASTER_SEM_DATASET.xlsx` ships a README claiming
"SEM-feasible: 0 countries", and its `Cross-Section 2023` sheet has zero
non-null values for every LPI sub-dimension. Spot-checking confirms that
sheet is a **strict year-2023 snapshot** of the panel for WGI/GDP/trade
variables (verified exactly against `CONSOLIDATED_panel_2010_2025.csv` for
USA, NGA, DEU, CHN, BRA), and 2023 was simply not a Logistics Performance
Index survey year (LPI was fielded in 2010, 2012, 2014, 2016, 2018 and 2022
only) -- so every LPI column in that sheet is empty by construction, not
because the data does not exist.

## Reconciliation rule applied here

For every indicator sourced from the long panel, each country's **most
recent non-null observation across 2010-2025** is taken, and the originating
year is recorded in a companion `<variable>_vintage_year` column, so every
value in `analytic_cross_section.csv` is auditable back to its source year.
AI-readiness, ITU digital-skills and Stanford HAI indicators are taken
directly from the pre-built wide cross-section sheet, since those sources
are already "latest-available" composites by construction (e.g.
`Avg_AI_Readiness` averages the Oxford Insights 2021-2025 report years).

World Bank regional/income aggregates (e.g. `WLD`, `SSF`, `HIC`, `LCN`, ...
48 aggregate codes identified via null `income_level`) are excluded, leaving
217 sovereign countries/territories.

## Dominant vintage year per panel indicator (top 3 by country count)

- `RnD_expenditure_pct_GDP`: 2022 (n=52), 2023 (n=27), 2021 (n=13)
- `Internet_users_pct_pop`: 2023 (n=132), 2024 (n=49), 2017 (n=13)
- `LPI_timeliness`: 2022 (n=138), 2018 (n=26), 2016 (n=3)
- `LPI_infrastructure`: 2022 (n=138), 2018 (n=26), 2016 (n=3)
- `LPI_tracking_tracing`: 2022 (n=138), 2018 (n=26), 2016 (n=3)
- `LPI_customs`: 2022 (n=138), 2018 (n=26), 2016 (n=3)
- `LPI_intl_shipments`: 2022 (n=138), 2018 (n=26), 2016 (n=3)
- `LPI_logistics_quality`: 2022 (n=138), 2018 (n=26), 2016 (n=3)
- `LPI_overall`: 2022 (n=138), 2018 (n=26), 2016 (n=3)
- `Exports_pct_GDP`: 2024 (n=160), 2023 (n=13), 2022 (n=9)
- `Trade_openness`: 2024 (n=160), 2023 (n=13), 2022 (n=9)
- `GDP_per_capita_const2015`: 2024 (n=192), 2023 (n=10), 2022 (n=6)
- `WGI_govt_effectiveness`: 2023 (n=205)
- `WGI_rule_of_law`: 2023 (n=205)
- `WGI_regulatory_quality`: 2023 (n=205)
- `WGI_control_corruption`: 2023 (n=205)

As expected, the LPI sub-dimensions resolve mostly to **2022** (the most
recent LPI survey), with a smaller number of countries falling back to 2018
or 2016 where 2022 data is missing. WGI and economic indicators resolve
almost entirely to the most recent panel years (2023-2024).

## Construct-level coverage

- **ξ1 AI Readiness** (Avg_AI_Readiness, Internet_users_pct_pop): 174 / 217 countries with all 2 indicators observed
- **η1 SC Performance** (LPI_timeliness, LPI_infrastructure, LPI_tracking_tracing, LPI_customs, LPI_intl_shipments, LPI_logistics_quality): 168 / 217 countries with all 6 indicators observed
- **η2 Economic Performance** (GDP_per_capita_const2015, Exports_pct_GDP, Trade_openness): 189 / 217 countries with all 3 indicators observed
- **ξ2 Institutional Quality** (WGI_govt_effectiveness, WGI_rule_of_law, WGI_regulatory_quality, WGI_control_corruption): 205 / 217 countries with all 4 indicators observed

**Countries with all four constructs fully observed (every indicator
present, no missing items): 141 / 217.**

This is the analytic sample used for the reliability, CFA/SEM, moderation
and mediation analyses in `reports/results_report.md`. It differs from the
original SPSS pilot's reported N (~164) because that pilot's composite
indices appear to have tolerated partial missingness within a construct
(e.g. averaging over whichever LPI sub-scores were non-null); here a
stricter listwise-complete rule is used per construct for comparability of
degrees of freedom across the regression and SEM stages.

## Construct definitions used

- **ξ1 AI Readiness**: Avg_AI_Readiness, Internet_users_pct_pop
  (`RnD_expenditure_pct_GDP` was tested as a candidate third item but dropped:
  it lowers Cronbach's alpha from 0.796 to 0.640 on the 2-item construct and,
  being globally sparse, shrinks the AI-readiness-complete sample from 174 to
  124 countries -- it is retained as a raw column in the dataset for
  robustness checks but excluded from the construct.)
- **η1 SC Performance**: LPI_timeliness, LPI_infrastructure, LPI_tracking_tracing, LPI_customs, LPI_intl_shipments, LPI_logistics_quality
  (`LPI_overall` is deliberately excluded -- it is an arithmetic composite of
  the six sub-dimensions already in the construct, so including it would be
  tautological/circular for reliability and CFA estimation.)
- **η2 Economic Performance**: GDP_per_capita_const2015, Exports_pct_GDP, Trade_openness
  (GDP per capita is log-transformed before standardisation in the analysis
  stage -- see `reports/results_report.md` -- to correct the severe positive
  skew/kurtosis the original SPSS pilot flagged but did not resolve.)
- **ξ2 Institutional Quality**: WGI_govt_effectiveness, WGI_rule_of_law, WGI_regulatory_quality, WGI_control_corruption
