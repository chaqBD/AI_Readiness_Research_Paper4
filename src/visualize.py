"""Figures for the results report. Static matplotlib PNGs following
the project's data-viz conventions: fixed-order categorical hues (never
cycled), one axis per chart, a legend whenever 2+ series are shown, a single
light-to-dark hue for sequential/magnitude encodings, and a two-hue-plus-
neutral-midpoint diverging scheme for signed/correlation values (no rainbow
colormaps).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("diverging_blue_red", ["#2a78d6", "#f0efec", "#e34948"])

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRIDLINE,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = "outputs/figures"


def distributions(composite_df: pd.DataFrame, labels: dict, path=f"{OUT}/construct_distributions.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (key, color) in zip(axes.flat, zip(composite_df.columns, CATEGORICAL)):
        s = composite_df[key].dropna()
        ax.hist(s, bins=18, color=color, edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(s.mean(), color=INK_PRIMARY, linewidth=1, linestyle="--")
        ax.set_title(labels[key], fontsize=11, color=INK_PRIMARY)
        ax.set_xlabel("Composite z-score")
        ax.set_ylabel("Countries")
        ax.grid(axis="y", linewidth=0.6)
    fig.suptitle("Distribution of Composite Constructs", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def econ_skew_fix(raw: pd.Series, logged: pd.Series, path=f"{OUT}/econ_skew_fix.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, s, title, color in zip(
        axes, [raw, logged],
        ["Before: GDP per capita (z-scored)", "After: log(GDP per capita), z-scored"],
        [CATEGORICAL[1], CATEGORICAL[0]],
    ):
        ax.hist(s.dropna(), bins=18, color=color, edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("z-score")
        ax.set_ylabel("Countries")
        ax.grid(axis="y", linewidth=0.6)
    fig.suptitle("Economic Performance: Resolving the Skew/Kurtosis Limitation", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def correlation_heatmap(corr: pd.DataFrame, path=f"{OUT}/correlation_heatmap.png"):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            text_color = "#ffffff" if abs(val) > 0.6 else INK_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson r", color=INK_PRIMARY)
    ax.set_title("Correlation Matrix of Composite Constructs", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coefficient_forest(coef_df: pd.DataFrame, path=f"{OUT}/path_coefficients.png"):
    """coef_df columns: label, estimate, ci_low, ci_high"""
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(coef_df) + 1.5))
    y = np.arange(len(coef_df))[::-1]
    for yi, (_, row) in zip(y, coef_df.iterrows()):
        color = CATEGORICAL[0] if row["estimate"] >= 0 else CATEGORICAL[7]
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=color, linewidth=2)
        ax.plot(row["estimate"], yi, "o", color=color, markersize=8)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(coef_df["label"], fontsize=10)
    ax.set_xlabel("Standardised coefficient (95% CI)")
    ax.set_title("Structural Path Coefficients", fontsize=12)
    ax.grid(axis="x", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def simple_slopes_plot(df: pd.DataFrame, model_full, x="AI_c", w="INST_c", y="SC", path=f"{OUT}/simple_slopes.png"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    w_sd = df[w].std()
    x_range = np.linspace(df[x].min(), df[x].max(), 100)
    levels = [("-1 SD Institutional Quality", -w_sd, CATEGORICAL[1]),
              ("Mean Institutional Quality", 0.0, CATEGORICAL[0]),
              ("+1 SD Institutional Quality", w_sd, CATEGORICAL[6])]
    b0 = model_full.params["const"]
    b_x = model_full.params[x]
    b_w = model_full.params[w]
    b_inter = model_full.params["interaction"]
    for label, w_val, color in levels:
        y_pred = b0 + b_x * x_range + b_w * w_val + b_inter * x_range * w_val
        ax.plot(x_range, y_pred, color=color, linewidth=2.2, label=label)
    ax.scatter(df[x], df[y], color=MUTED, alpha=0.35, s=18, zorder=0)
    ax.set_xlabel("AI Readiness (centered composite)")
    ax.set_ylabel("SC Performance (composite z-score)")
    ax.set_title("Simple Slopes: AI Readiness -> SC Performance\nby Institutional Quality", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def johnson_neyman_plot(df: pd.DataFrame, model_full, jn_bounds, x="AI_c", w="INST_c", path=f"{OUT}/johnson_neyman.png"):
    from scipy import stats as sstats

    w_range = np.linspace(df[w].min(), df[w].max(), 300)
    b_x = model_full.params[x]
    b_inter = model_full.params["interaction"]
    cov = model_full.cov_params()
    var_bx, var_inter = cov.loc[x, x], cov.loc["interaction", "interaction"]
    cov_bx_inter = cov.loc[x, "interaction"]
    df_resid = model_full.df_resid
    t_crit = sstats.t.ppf(0.975, df_resid)

    slopes = b_x + w_range * b_inter
    se = np.sqrt(var_bx + 2 * w_range * cov_bx_inter + w_range ** 2 * var_inter)
    ci_lo, ci_hi = slopes - t_crit * se, slopes + t_crit * se

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(w_range, slopes, color=CATEGORICAL[0], linewidth=2.2, label="Conditional slope of AI -> SC")
    ax.fill_between(w_range, ci_lo, ci_hi, color=CATEGORICAL[0], alpha=0.15, label="95% CI")
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    if jn_bounds:
        for b in jn_bounds:
            if w_range.min() <= b <= w_range.max():
                ax.axvline(b, color=CATEGORICAL[7], linewidth=1.5, linestyle=":")
    ax.set_xlabel("Institutional Quality (centered composite)")
    ax.set_ylabel("Conditional slope: AI Readiness -> SC Performance")
    ax.set_title("Johnson-Neyman Region of Significance", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
