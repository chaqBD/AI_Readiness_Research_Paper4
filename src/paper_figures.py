"""Novel figures for the Q1 paper: conceptual framework, regional geography,
multidimensional bubble chart, and a combined moderation panel. Reuses the
same construct definitions and palette as the rest of the pipeline."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, "src")
from constructs import load_analytic_sample, composite_scores, LABELS

OUT = "outputs/figures/paper"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

REGION_NAMES = {
    "EAS": "East Asia & Pacific",
    "ECS": "Europe & Central Asia",
    "LCN": "Latin America & Caribbean",
    "MEA": "Middle East & North Africa",
    "NAC": "North America",
    "SAS": "South Asia",
    "SSF": "Sub-Saharan Africa",
}
REGION_COLOR = {
    "EAS": BLUE, "ECS": ORANGE, "LCN": AQUA, "MEA": YELLOW,
    "NAC": MAGENTA, "SAS": GREEN, "SSF": VIOLET,
}
REGION_ORDER = ["EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
})


def fig1_conceptual_framework():
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(-2.3, 8.6)
    ax.axis("off")

    boxes = {
        "AI": (1.5, 5.6, "ξ₁ AI Readiness\n(Oxford Insights,\nInternet users)", BLUE),
        "INST": (1.5, 2.2, "ξ₂ Institutional\nQuality\n(4 WGI indicators)", VIOLET),
        "SC": (5.6, 3.9, "η₁ SC Performance\n(6 LPI sub-dimensions)", AQUA),
        "ECON": (9.4, 3.9, "η₂ Economic\nPerformance\n(log-GDP, exports,\ntrade openness)", ORANGE),
    }
    w, h = 2.5, 1.55
    centers = {}
    for key, (cx, cy, label, color) in boxes.items():
        box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                              boxstyle="round,pad=0.02,rounding_size=0.12",
                              linewidth=1.6, edgecolor=color, facecolor="#ffffff", zorder=3)
        ax.add_patch(box)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5,
                 color=INK, zorder=4, linespacing=1.3, fontweight="medium")
        centers[key] = (cx, cy, w, h)

    def edge_point(key, dx, dy):
        cx, cy, bw, bh = centers[key]
        return (cx + dx * bw / 2, cy + dy * bh / 2)

    def arrow(p0, p1, color, style="-", lw=2.0, curve=0.0, z=2):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                             linewidth=lw, color=color, linestyle=style,
                             connectionstyle=f"arc3,rad={curve}", zorder=z)
        ax.add_patch(a)

    # H1a: AI -> SC
    arrow(edge_point("AI", 1, -0.3), edge_point("SC", -1, 0.55), BLUE, lw=2.4, curve=-0.08)
    ax.text(3.75, 5.35, "H1a (+)", color=BLUE, fontsize=10.5, fontweight="bold")

    # H1b: INST -> SC
    arrow(edge_point("INST", 1, 0.3), edge_point("SC", -1, -0.55), VIOLET, lw=2.4, curve=0.08)
    ax.text(3.75, 2.65, "H1b (+)", color=VIOLET, fontsize=10.5, fontweight="bold")

    # H1c: moderation (dashed, into the AI->SC path)
    mod_x, mod_y = 3.75, 4.6
    ax.plot([mod_x], [mod_y], marker="o", markersize=7, color=RED, zorder=5)
    a = FancyArrowPatch((2.75, 3.1), (mod_x, mod_y - 0.18), arrowstyle="-|>",
                         mutation_scale=16, linewidth=1.8, color=RED, linestyle="--",
                         connectionstyle="arc3,rad=-0.25", zorder=4)
    ax.add_patch(a)
    ax.text(1.75, 4.05, "H1c: moderation\n(reversed sign —\nsee Figure 6)", color=RED,
            fontsize=9.2, fontweight="bold", linespacing=1.3)

    # SC -> ECON (b path) and AI -> ECON direct (c' path)
    arrow(edge_point("SC", 1, -0.1), edge_point("ECON", -1, -0.1), AQUA, lw=2.2)
    ax.text(7.5, 3.55, "b (n.s.)", color=AQUA, fontsize=10, fontweight="bold")

    arrow(edge_point("AI", 1, -0.1), edge_point("ECON", -1, 0.9), ORANGE, lw=2.2, curve=-0.2)
    ax.text(6.3, 6.1, "c′ (direct, +)", color=ORANGE, fontsize=10, fontweight="bold")

    ax.text(5.5, 8.0, "Conceptual and Hypothesised Structural Model", ha="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(5.5, 7.5, "Mediation (H2a): does η₁ SC Performance carry ξ₁ AI Readiness's effect on\n"
                      "η₂ Economic Performance via the a×b pathway? Result: not supported.",
            ha="center", fontsize=9.6, color=INK2, style="italic", linespacing=1.35)

    legend_handles = [
        mpatches.Patch(color=BLUE, label="H1a: AI Readiness → SC Performance"),
        mpatches.Patch(color=VIOLET, label="H1b: Institutional Quality → SC Performance"),
        mpatches.Patch(color=RED, label="H1c: Institutions × AI interaction (moderation)"),
        mpatches.Patch(color=AQUA, label="H2a, b-path: SC Performance → Economic Performance"),
        mpatches.Patch(color=ORANGE, label="H2a, c′-path: AI Readiness → Economic Performance (direct)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
              ncol=1, frameon=False, fontsize=9.2, labelcolor=INK2, handlelength=1.4,
              handleheight=1.0, labelspacing=0.9)

    fig.savefig(f"{OUT}/fig1_conceptual_framework.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


def fig2_regional_comparison(complete):
    scores = composite_scores(complete)
    scores["region"] = complete["region"].values
    scores = scores[scores["region"].isin(REGION_ORDER)]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5.2), sharey=True)
    constructs = ["AI", "SC", "INST", "ECON"]
    titles = [LABELS[c] for c in constructs]

    for ax, construct, title in zip(axes, constructs, titles):
        data = [scores.loc[scores["region"] == r, construct].dropna().values for r in REGION_ORDER]
        medians = [np.median(d) for d in data]
        order = np.argsort(medians)
        ordered_regions = [REGION_ORDER[i] for i in order]
        ordered_data = [data[i] for i in order]
        bp = ax.boxplot(ordered_data, vert=False, patch_artist=True, widths=0.6,
                         medianprops=dict(color=INK, linewidth=1.6),
                         whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                         flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                          markeredgecolor="none", alpha=0.6))
        for patch, r in zip(bp["boxes"], ordered_regions):
            patch.set_facecolor(REGION_COLOR[r])
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK)
            patch.set_linewidth(0.8)
        ax.set_yticks(range(1, len(ordered_regions) + 1))
        ax.set_yticklabels([REGION_NAMES[r] for r in ordered_regions], fontsize=9.5)
        ax.axvline(0, color=GRID, linewidth=1, zorder=0)
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("Composite z-score", fontsize=9.5, color=INK2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(length=0)

    fig.suptitle("Regional Distribution of the Four Composite Constructs (N = 141)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_regional_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


def fig3_choropleth_small_multiples(complete):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scores = composite_scores(complete)
    scores["country_code"] = complete["country_code"].values
    scores["country_name"] = complete["country_name_wb"].values

    constructs = ["AI", "SC", "INST", "ECON"]
    titles = [LABELS[c] for c in constructs]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}],
               [{"type": "choropleth"}, {"type": "choropleth"}]],
        subplot_titles=titles, horizontal_spacing=0.02, vertical_spacing=0.08,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    blue_scale = [[0, SEQ_BLUE[0]], [0.17, SEQ_BLUE[1]], [0.33, SEQ_BLUE[2]],
                  [0.5, SEQ_BLUE[3]], [0.67, SEQ_BLUE[4]], [0.83, SEQ_BLUE[5]], [1.0, SEQ_BLUE[6]]]

    for construct, (r, c) in zip(constructs, positions):
        fig.add_trace(
            go.Choropleth(
                locations=scores["country_code"], z=scores[construct],
                locationmode="ISO-3", colorscale=blue_scale,
                marker_line_color="white", marker_line_width=0.3,
                colorbar=dict(title="z", len=0.42, thickness=12,
                               x=1.0 if c == 2 else 0.47,
                               y=0.8 if r == 1 else 0.19),
                text=scores["country_name"], hovertemplate="%{text}: %{z:.2f}<extra></extra>",
            ),
            row=r, col=c,
        )

    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                     bgcolor=SURFACE, landcolor="#efeeea", showcountries=False)
    fig.update_layout(
        title=dict(text="Cross-National Geography of the Four Composite Constructs (N = 141)",
                    x=0.5, font=dict(size=16, color=INK)),
        paper_bgcolor=SURFACE, font=dict(color=INK2, size=12),
        margin=dict(l=10, r=10, t=90, b=10), height=640, width=1150,
    )
    fig.write_image(f"{OUT}/fig3_choropleth.png", scale=2)
    print("fig3 done")


def fig4_bubble_chart(complete):
    scores = composite_scores(complete)
    scores["region"] = complete["region"].values
    scores["country_name"] = complete["country_name_wb"].values
    pop = pd.read_excel("Dataset/CONSOLIDATED_MASTER_SEM_DATASET.xlsx",
                         sheet_name="Cross-Section 2023", header=3)[["country_code", "Population_total"]]
    merged = scores.merge(pd.DataFrame({"country_code": complete["country_code"].values}),
                           left_index=True, right_index=True)
    merged = merged.merge(pop, on="country_code", how="left")
    merged = merged[merged["region"].isin(REGION_ORDER)].dropna(subset=["Population_total"])

    fig, ax = plt.subplots(figsize=(10, 7.2))
    size = 40 + 2200 * (np.log10(merged["Population_total"]) - np.log10(merged["Population_total"]).min()) / \
        (np.log10(merged["Population_total"]).max() - np.log10(merged["Population_total"]).min())

    for r in REGION_ORDER:
        mask = merged["region"] == r
        ax.scatter(merged.loc[mask, "AI"], merged.loc[mask, "SC"], s=size[mask],
                    color=REGION_COLOR[r], alpha=0.68, edgecolor="white", linewidth=0.8,
                    label=REGION_NAMES[r], zorder=3)

    highlight = merged.nlargest(6, "Population_total")
    for _, row in highlight.iterrows():
        ax.annotate(row["country_name"], (row["AI"], row["SC"]), fontsize=8.3,
                     color=INK2, xytext=(5, 4), textcoords="offset points")

    z = np.polyfit(merged["AI"], merged["SC"], 1)
    xs = np.linspace(merged["AI"].min(), merged["AI"].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), color=INK, linewidth=1.4, linestyle="--", alpha=0.6, zorder=2)

    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1, zorder=0)
    ax.set_xlabel("\u03be\u2081 AI Readiness (composite z-score)", fontsize=11, color=INK)
    ax.set_ylabel("\u03b7\u2081 SC Performance (composite z-score)", fontsize=11, color=INK)
    ax.set_title("AI Readiness, Supply-Chain Performance, Region and Population (N = 139)",
                 fontsize=12.5, fontweight="bold", color=INK, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, title="World Bank region",
                     title_fontsize=9.5, labelcolor=INK2)
    ax.text(0.98, 0.03, "Bubble size \u221d population (log scale)", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=MUTED, style="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_bubble_region_population.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


def fig5_moderation_panel(complete):
    from moderation import fit_moderation, simple_slopes, johnson_neyman

    scores = composite_scores(complete)
    d, model_main, model_full, vif, delta_r2 = fit_moderation(scores)
    w_mean, w_sd = scores["INST"].mean(), scores["INST"].std(ddof=1)
    slopes = simple_slopes(model_full, w_mean, w_sd)
    jn = johnson_neyman(model_full)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

    ax1.scatter(d["AI_c"], d["SC"], s=22, color=MUTED, alpha=0.35, edgecolor="none", zorder=1)
    xs = np.linspace(d["AI_c"].min(), d["AI_c"].max(), 100)
    b_x = model_full.params["AI_c"]
    b_inter = model_full.params["interaction"]
    b0 = model_full.params["const"]
    b_w = model_full.params["INST_c"]
    line_specs = [("-1 SD Institutions", -w_sd, ORANGE), ("Mean Institutions", 0.0, BLUE),
                  ("+1 SD Institutions", w_sd, VIOLET)]
    for label, w_val, color in line_specs:
        ys = b0 + b_x * xs + b_w * w_val + b_inter * xs * w_val
        ax1.plot(xs, ys, color=color, linewidth=2.6, label=label, zorder=3)
    ax1.set_xlabel("ξ₁ AI Readiness (centered composite)", fontsize=10.5)
    ax1.set_ylabel("η₁ SC Performance (composite z-score)", fontsize=10.5)
    ax1.set_title("(a) Simple slopes by institutional quality", fontsize=11.5, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(MUTED)

    w_range = np.linspace(d["INST_c"].min() - 0.3, d["INST_c"].max() + 0.3, 200)
    cov = model_full.cov_params()
    var_bx, var_inter = cov.loc["AI_c", "AI_c"], cov.loc["interaction", "interaction"]
    cov_bx_inter = cov.loc["AI_c", "interaction"]
    df_resid = model_full.df_resid
    from scipy import stats as sstats
    t_crit = sstats.t.ppf(0.975, df_resid)
    cond_slope = b_x + w_range * b_inter
    se = np.sqrt(var_bx + 2 * w_range * cov_bx_inter + w_range ** 2 * var_inter)
    lo, hi = cond_slope - t_crit * se, cond_slope + t_crit * se

    ax2.plot(w_range, cond_slope, color=BLUE, linewidth=2.4, label="Conditional slope: AI → SC", zorder=3)
    ax2.fill_between(w_range, lo, hi, color=SEQ_BLUE[1], alpha=0.55, label="95% CI", zorder=2)
    ax2.axhline(0, color=INK2, linewidth=1.1, linestyle="--", zorder=1)
    if jn is not None:
        for bound in jn:
            if w_range.min() <= bound <= w_range.max():
                ax2.axvline(bound, color=RED, linewidth=1.6, linestyle=":", zorder=4)
                ax2.text(bound, ax2.get_ylim()[1] if False else cond_slope.max() * 0.92,
                         f"  JN = {bound:.2f}", color=RED, fontsize=9, fontweight="bold")
    ax2.set_xlabel("ξ₂ Institutional Quality (centered composite)", fontsize=10.5)
    ax2.set_ylabel("Conditional slope of AI Readiness → SC Performance", fontsize=10.5)
    ax2.set_title("(b) Johnson-Neyman region of significance", fontsize=11.5, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(MUTED)

    fig.suptitle("Institutional Quality Strengthens, Rather Than Weakens, the AI → SC Performance Effect",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_moderation_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig6 (moderation panel) done")


def main():
    full, complete = load_analytic_sample()
    fig1_conceptual_framework()
    fig2_regional_comparison(complete)
    fig3_choropleth_small_multiples(complete)
    fig4_bubble_chart(complete)
    fig5_moderation_panel(complete)


if __name__ == "__main__":
    main()
