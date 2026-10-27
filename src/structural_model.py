"""Full structural SEM (CFA + directional paths) and latent factor-score extraction."""
import pandas as pd
import semopy

STRUCTURAL_MODEL = """
AI =~ Avg_AI_Readiness + Internet_users_pct_pop
SC =~ LPI_timeliness + LPI_infrastructure + LPI_tracking_tracing + LPI_customs + LPI_intl_shipments + LPI_logistics_quality
ECON =~ log_GDP_per_capita + Exports_pct_GDP + Trade_openness
INST =~ WGI_govt_effectiveness + WGI_rule_of_law + WGI_regulatory_quality + WGI_control_corruption

SC ~ AI + INST
ECON ~ AI + SC
"""


def fit_structural_model(df: pd.DataFrame):
    model = semopy.Model(STRUCTURAL_MODEL)
    model.fit(df)
    return model


def structural_paths(model: semopy.Model) -> pd.DataFrame:
    ins = model.inspect(std_est=True)
    paths = ins[(ins["op"] == "~") & (ins["rval"].isin(["AI", "SC", "INST"])) & (ins["lval"].isin(["SC", "ECON"]))].copy()
    paths = paths.rename(columns={"lval": "outcome", "rval": "predictor", "Est. Std": "beta", "Estimate": "B"})
    paths["beta"] = paths["beta"].astype(float)
    return paths[["outcome", "predictor", "B", "beta", "Std. Err", "z-value", "p-value"]]


def factor_scores(model: semopy.Model, df: pd.DataFrame) -> pd.DataFrame:
    scores = model.predict_factors(df)
    return pd.DataFrame(scores, index=df.index)
