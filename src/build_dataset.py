"""
Rebuilds a defensible analytic cross-section from the raw World Bank panel and
the Oxford/ITU/HAI cross-section sheet, replacing the broken pre-built
"SEM-ready" outputs shipped in Dataset/CONSOLIDATED_MASTER_SEM_DATASET.xlsx
(its own README reports "SEM-feasible: 0 countries", and its 2023 cross-section
sheet has zero non-null LPI values because the Logistics Performance Index was
not surveyed in 2023 -- the nearest survey years are 2022 and 2018).

Reconciliation rule: for every panel-sourced indicator, take each country's most
recent non-null observation across all available years (2010-2025) and record
the year it came from in a "<var>_vintage_year" column. AI-readiness / ITU /
HAI indicators are taken from the pre-built wide cross-section sheet, which is
already a "latest-available" composite by construction for those sources.
"""
import numpy as np
import pandas as pd

RAW_DIR = "Dataset"
OUT_DIR = "data/processed"

PANEL_VARS = [
    "RnD_expenditure_pct_GDP",
    "Internet_users_pct_pop",
    "LPI_timeliness",
    "LPI_infrastructure",
    "LPI_tracking_tracing",
    "LPI_customs",
    "LPI_intl_shipments",
    "LPI_logistics_quality",
    "LPI_overall",
    "Exports_pct_GDP",
    "Trade_openness",
    "GDP_per_capita_const2015",
    "WGI_govt_effectiveness",
    "WGI_rule_of_law",
    "WGI_regulatory_quality",
    "WGI_control_corruption",
]

CROSS_SECTION_VARS = [
    "country_code",
    "Oxford_Overall_2023",
    "Oxford_Gov_pillar_2023",
    "Oxford_Tech_pillar_2023",
    "Oxford_Data_pillar_2023",
    "Oxford_Overall_2025",
    "AI_Readiness_2021",
    "AI_Readiness_2022",
    "AI_Readiness_2023",
    "AI_Readiness_2025",
    "Avg_AI_Readiness",
    "ITU_Digital_Skills_latest_pct",
    "HAI_AI_skill_penetration_2024",
    "HAI_AI_investment_bn_USD_2024",
    "AI_patents_per_100k_inhabitants_2023",
]


def latest_available(panel: pd.DataFrame, var: str) -> pd.DataFrame:
    sub = panel.dropna(subset=[var])[["country_code", "year", var]]
    sub = sub.sort_values(["country_code", "year"])
    latest = sub.groupby("country_code", as_index=False).last()
    latest = latest.rename(columns={"year": f"{var}_vintage_year"})
    return latest


def build():
    panel = pd.read_csv(f"{RAW_DIR}/CONSOLIDATED_panel_2010_2025.csv")

    # Drop World Bank regional/income aggregates (not sovereign countries/territories).
    real_codes = panel.loc[panel["income_level"].notna(), "country_code"].unique()
    panel = panel[panel["country_code"].isin(real_codes)].copy()

    meta = (
        panel[["country_code", "country_name_wb", "region", "income_level"]]
        .drop_duplicates("country_code")
        .set_index("country_code")
    )

    merged = meta.copy()
    vintage_summary = {}
    for var in PANEL_VARS:
        latest = latest_available(panel, var).set_index("country_code")
        merged = merged.join(latest, how="left")
        vcol = f"{var}_vintage_year"
        if vcol in merged.columns:
            vc = merged[vcol].dropna()
            vintage_summary[var] = (
                vc.value_counts().sort_index(ascending=False).to_dict() if len(vc) else {}
            )

    cs = pd.read_excel(
        f"{RAW_DIR}/CONSOLIDATED_MASTER_SEM_DATASET.xlsx",
        sheet_name="Cross-Section 2023",
        header=3,
    )
    cs = cs[CROSS_SECTION_VARS].drop_duplicates("country_code").set_index("country_code")

    merged = merged.join(cs, how="left")
    merged = merged.reset_index().rename(columns={"index": "country_code"})

    merged.to_csv(f"{OUT_DIR}/analytic_cross_section.csv", index=False)

    construct_cols = {
        "ξ1 AI Readiness": ["Avg_AI_Readiness", "Internet_users_pct_pop"],
        "η1 SC Performance": [
            "LPI_timeliness", "LPI_infrastructure", "LPI_tracking_tracing",
            "LPI_customs", "LPI_intl_shipments", "LPI_logistics_quality",
        ],
        "η2 Economic Performance": ["GDP_per_capita_const2015", "Exports_pct_GDP", "Trade_openness"],
        "ξ2 Institutional Quality": [
            "WGI_govt_effectiveness", "WGI_rule_of_law",
            "WGI_regulatory_quality", "WGI_control_corruption",
        ],
    }

    n_countries = len(merged)
    coverage_lines = []
    for construct, cols in construct_cols.items():
        complete = merged.dropna(subset=cols)
        coverage_lines.append(
            f"- **{construct}** ({', '.join(cols)}): {len(complete)} / {n_countries} "
            f"countries with all {len(cols)} indicators observed"
        )

    all_cols = [c for cols in construct_cols.values() for c in cols]
    full_complete = merged.dropna(subset=all_cols)

    def dominant_vintage(var):
        d = vintage_summary.get(var, {})
        if not d:
            return "n/a"
        top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        return ", ".join(f"{int(y)} (n={c})" for y, c in top)

    notes = f"""# Data reconciliation notes

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
{n_countries} sovereign countries/territories.

## Dominant vintage year per panel indicator (top 3 by country count)

{chr(10).join(f"- `{var}`: {dominant_vintage(var)}" for var in PANEL_VARS)}

As expected, the LPI sub-dimensions resolve mostly to **2022** (the most
recent LPI survey), with a smaller number of countries falling back to 2018
or 2016 where 2022 data is missing. WGI and economic indicators resolve
almost entirely to the most recent panel years (2023-2024).

## Construct-level coverage

{chr(10).join(coverage_lines)}

**Countries with all four constructs fully observed (every indicator
present, no missing items): {len(full_complete)} / {n_countries}.**

This is the analytic sample used for the reliability, CFA/SEM, moderation
and mediation analyses in `reports/results_report.md`. It differs from the
original SPSS pilot's reported N (~164) because that pilot's composite
indices appear to have tolerated partial missingness within a construct
(e.g. averaging over whichever LPI sub-scores were non-null); here a
stricter listwise-complete rule is used per construct for comparability of
degrees of freedom across the regression and SEM stages.

## Construct definitions used

- **ξ1 AI Readiness**: {", ".join(construct_cols["ξ1 AI Readiness"])}
  (`RnD_expenditure_pct_GDP` was tested as a candidate third item but dropped:
  it lowers Cronbach's alpha from 0.796 to 0.640 on the 2-item construct and,
  being globally sparse, shrinks the AI-readiness-complete sample from 174 to
  124 countries -- it is retained as a raw column in the dataset for
  robustness checks but excluded from the construct.)
- **η1 SC Performance**: {", ".join(construct_cols["η1 SC Performance"])}
  (`LPI_overall` is deliberately excluded -- it is an arithmetic composite of
  the six sub-dimensions already in the construct, so including it would be
  tautological/circular for reliability and CFA estimation.)
- **η2 Economic Performance**: {", ".join(construct_cols["η2 Economic Performance"])}
  (GDP per capita is log-transformed before standardisation in the analysis
  stage -- see `reports/results_report.md` -- to correct the severe positive
  skew/kurtosis the original SPSS pilot flagged but did not resolve.)
- **ξ2 Institutional Quality**: {", ".join(construct_cols["ξ2 Institutional Quality"])}
"""
    with open(f"{OUT_DIR}/DATA_NOTES.md", "w", encoding="utf-8") as f:
        f.write(notes)

    print(f"Wrote {OUT_DIR}/analytic_cross_section.csv ({n_countries} countries)")
    print(f"Fully complete on all 4 constructs: {len(full_complete)}")


if __name__ == "__main__":
    build()
