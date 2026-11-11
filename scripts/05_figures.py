"""Figures 6-13 of the manuscript, from the de-identified data. Run after 02_count_models.py, whose
Mancl-DeRouen p-values (output/results_values.csv) set the significance marks in Figs 8, 12 and 13.
Output file names match the manuscript's figure files (output/figures/)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import FIGURES, OUT, load_force, load_navigation  # noqa: E402

ORDER = ["Haptic Off", "Haptic On", "Joystick", "Manual"]  # alphabetical, as in the published figures
SPLIT = ["t_BCA_s", "t_CCA_s", "t_ICA_s"]
STEP_COLORS = {"BCA": "#5DADE2", "CCA": "#F39C12", "ICA": "#27AE60"}
TAB10 = plt.get_cmap("tab10").colors
EXPERT_COL, NOVICE_COL = TAB10[0], TAB10[1]

nav = load_navigation()
force = load_force("peak_force_mN")
force["force_N"] = force["force"] / 1000.0  # figures are in N, as published
res = pd.read_csv(os.path.join(OUT, "results_values.csv")).set_index("key")["value"]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def save(fig, name, dpi, tight=True):
    fig.savefig(os.path.join(FIGURES, name), dpi=dpi, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print("  wrote", name)


def stacked_split(ax, x, means, sds, width=0.8):
    """Stacked BCA/CCA/ICA segments; error bars = SD of the cumulative checkpoint times."""
    bca = means["t_BCA_s"]
    cca = means["t_CCA_s"] - means["t_BCA_s"]
    ica = means["t_ICA_s"] - means["t_CCA_s"]
    ax.bar(x, bca, width=width, color=STEP_COLORS["BCA"], label="BCA")
    ax.bar(x, cca, width=width, bottom=bca, color=STEP_COLORS["CCA"], label="CCA")
    ax.bar(x, ica, width=width, bottom=bca + cca, color=STEP_COLORS["ICA"], label="ICA")
    off = 0.15
    ax.errorbar(x - off, bca, yerr=sds["t_BCA_s"], fmt="none", ecolor="#333333", capsize=5, linewidth=2)
    ax.errorbar(x, bca + cca, yerr=sds["t_CCA_s"], fmt="none", ecolor="#333333", capsize=5, linewidth=2)
    ax.errorbar(x + off, bca + cca + ica, yerr=sds["t_ICA_s"], fmt="none", ecolor="black", capsize=5, linewidth=2)


# ---------------------------------------------------------------- Fig 6: split times by controller
with plt.rc_context({"font.size": 30, "ytick.labelsize": 30, "xtick.labelsize": 34, "axes.labelsize": 30}):
    g = nav.groupby("controller", observed=True)[SPLIT]
    m, s = g.mean().reindex(ORDER), g.std().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(ORDER))
    stacked_split(ax, x, {k: m[k].to_numpy() for k in SPLIT}, {k: s[k].to_numpy() for k in SPLIT})
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=30)
    ax.set_ylabel("Time (s)")
    h, _ = ax.get_legend_handles_labels()
    h.append(plt.Line2D([0], [0], color="black", lw=0, label="Error bars show standard deviation"))
    ax.legend(handles=h, title="Step", fontsize=14, title_fontsize=16)
    fig.tight_layout()
    save(fig, "Figs_controller_plot_split_times_stacked_bar_figure.png", 200)

# ---------------------------------------------------------------- Fig 9: split times, expert vs novice
with plt.rc_context({"font.size": 30, "ytick.labelsize": 30, "axes.labelsize": 30}):
    groups = ["Combined"] + ORDER
    bw, pair_gap, group_gap = 0.35, 0.10, 0.35
    xs, means, sds, labels = [], {k: [] for k in SPLIT}, {k: [] for k in SPLIT}, []
    for i, grp in enumerate(groups):
        base = i * (2 * bw + pair_gap + group_gap)
        for j, e in enumerate(["Expert", "Novice"]):
            sub = nav[nav.expertise == e] if grp == "Combined" else nav[(nav.expertise == e) & (nav.controller == grp)]
            for k in SPLIT:
                means[k].append(sub[k].mean())
                sds[k].append(sub[k].std())
            xs.append(base + j * (bw + pair_gap))
            labels.append(e)
    xs = np.array(xs)
    fig, ax = plt.subplots(figsize=(15, 8.6))
    stacked_split(ax, xs, {k: np.array(v) for k, v in means.items()}, {k: np.array(v) for k, v in sds.items()}, width=bw)
    ax.set_ylim(-70, 600)
    for xp, lab in zip(xs, labels):
        ax.text(xp, -70 + 8, lab, ha="center", va="bottom", fontsize=16)
    ax.set_xticks([xs[2 * i:2 * i + 2].mean() for i in range(len(groups))])
    ax.set_xticklabels(groups, fontsize=30)
    ax.set_ylabel("Time (s)")
    ax.legend(title="Step", fontsize=26, title_fontsize=30, loc="upper right")
    fig.tight_layout()
    save(fig, "Figs_novice_vs_expert_combined_plot_group_split_times_stacked_bar_combined_figure.png", 200)


# ---------------------------------------------------------------- Fig 7: per-sensor force by controller
def grouped_boxes(ax, data, n_groups, colors, alpha, width_total=0.8):
    """data[j][s] = values of group j at sensor s; one box per group per sensor."""
    width = width_total / n_groups
    centers = np.arange(8)
    for j in range(n_groups):
        bp = ax.boxplot(data[j], positions=centers + (j - (n_groups - 1) / 2) * width, widths=width * 0.9,
                        patch_artist=True, manage_ticks=False, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[j])
            patch.set_alpha(alpha)
        for med in bp["medians"]:
            med.set_color("black")
    ax.set_xticks(centers)


fig, ax = plt.subplots(figsize=(10, 6))
data = [[force.loc[(force.controller == c) & (force.sensor == s), "force_N"].to_numpy() for s in range(1, 9)]
        for c in ORDER]
grouped_boxes(ax, data, len(ORDER), TAB10, 0.8)
ax.set_xticklabels([f"Sensor {s}" for s in range(1, 9)], rotation=90, fontsize=13)
ax.tick_params(axis="y", labelsize=14)
ax.set_xlabel("Sensor", fontsize=16)
ax.set_ylabel("max force (N)", fontsize=16)
ax.legend([plt.Line2D([0], [0], color=TAB10[i], lw=6, alpha=0.8) for i in range(len(ORDER))], ORDER,
          title="controller", loc="upper right", fontsize=14, title_fontsize=16, frameon=False)
fig.tight_layout()
save(fig, "Figs_controller_grouped_boxplots_by_controller.png", 150, tight=False)

# ---------------------------------------------------------------- Fig 10: per-sensor force, expert vs novice
fig, ax = plt.subplots(figsize=(8, 3))
data = [[force.loc[(force.expertise == e) & (force.sensor == s), "force_N"].to_numpy() for s in range(1, 9)]
        for e in ["Expert", "Novice"]]
grouped_boxes(ax, data, 2, [EXPERT_COL, NOVICE_COL], 0.9)
ax.set_xticklabels([f"sensor {s}" for s in range(1, 9)], rotation=90, fontsize=8)
ax.set_ylabel("max force (N)", fontsize=14)
ax.legend([plt.Line2D([0], [0], color=c, lw=6, alpha=0.9) for c in (EXPERT_COL, NOVICE_COL)], ["expert", "novice"],
          title="group", loc="upper right", fontsize=14)
fig.tight_layout()
save(fig, "Figs_novice_vs_expert_novice_vs_expert_boxplot.png", 300, tight=False)

# ---------------------------------------------------------------- Fig 11: per sensor x controller x expertise
fig, axes = plt.subplots(4, 2, figsize=(10, 11))
for s, ax in zip(range(1, 9), axes.ravel()):
    centers = np.arange(len(ORDER))
    for e, col, off in (("Expert", EXPERT_COL, -0.18), ("Novice", NOVICE_COL, 0.18)):
        vals = [force.loc[(force.sensor == s) & (force.controller == c) & (force.expertise == e), "force_N"].to_numpy()
                for c in ORDER]
        bp = ax.boxplot(vals, positions=centers + off, widths=0.32, patch_artist=True, showfliers=False,
                        manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.9)
        for med in bp["medians"]:
            med.set_color("black")
    ax.set_xticks(centers)
    ax.set_xticklabels(ORDER, rotation=45, fontsize=11)
    ax.set_title(f"Sensor {s}", fontsize=12)
    ax.set_ylabel("max force (N)", fontsize=12)
    ax.set_ylim(0, 0.07)
    if s == 1:
        ax.legend([plt.Line2D([0], [0], color=c, lw=6) for c in (EXPERT_COL, NOVICE_COL)], ["expert", "novice"],
                  loc="upper right", fontsize=11, frameon=False)
fig.tight_layout()
save(fig, "Figs_novice_vs_expert_by_controller_all_channels_by_controller_and_expertise.png", 150, tight=False)

# ---------------------------------------------------------------- Figs 8, 12, 13 (style of R1's regen_figures.py)
BAR_STYLE = {"font.size": 30, "axes.labelsize": 34, "xtick.labelsize": 28, "ytick.labelsize": 28}
BLUE, GREEN = "#7DBDE7", "#52BE80"
EXPERT_BLUE, NOVICE_ORANGE = "#4C9BD4", "#F0A02A"


def bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.0, c="k")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=30)


def controller_bars(col, short, ylabel, color, ylim, name):
    g = nav.groupby("controller", observed=True)[col].agg(["mean", "std"]).reindex(ORDER)
    with plt.rc_context(BAR_STYLE):
        fig, ax = plt.subplots(figsize=(11, 11))
        x = np.arange(len(ORDER))
        ax.bar(x, g["mean"], yerr=g["std"], color=color, error_kw={"lw": 2.5, "ecolor": "black"})
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, rotation=45, ha="right")
        ax.set_ylim(*ylim)
        top = g["mean"] + g["std"]
        # robotic vs manual: star above each robotic bar if p < 0.05 (Mancl-DeRouen GEE)
        for c in ["Haptic Off", "Haptic On", "Joystick"]:
            p = res[f"{short}.gee_md.{c}_vs_Manual.p"]
            if p < 0.05 and short == "prolapse":
                ax.text(ORDER.index(c), top[c] + 0.08, stars(p), ha="center", va="bottom", fontsize=30)
            elif p < 0.05:  # catheterizations: the one significant contrast is drawn as a bracket to Manual
                bracket(ax, ORDER.index(c), ORDER.index("Manual"), top[c] + 0.15, 0.12, stars(p))
        # robotic vs joystick brackets, only where p < 0.05
        level = 0
        for c in ["Haptic On", "Haptic Off"]:
            p = res[f"{short}.gee_md.{c}_vs_Joystick.p"]
            if p < 0.05:
                y = max(top[c], top["Joystick"]) + 0.45 + 0.35 * level
                bracket(ax, ORDER.index(c), ORDER.index("Joystick"), y, 0.15, stars(p))
                level += 1
        fig.tight_layout()
        save(fig, name, 170, tight=False)


controller_bars("incorrect_catheterizations", "cath", "Incorrect Catheterizations", BLUE, (-0.5, 3.0),
                "Figs_controller_plot_catheterisations_bar_figure.png")
controller_bars("prolapse_events", "prolapse", "Number of Prolapse Events", GREEN, (-0.5, 3.3),
                "Figs_controller_plot_prolapse_bar_figure.png")


def expertise_bars(col, short, ylabel, y_max, bracket_h, name):
    groups = ["Combined", "Haptic On", "Haptic Off", "Joystick", "Manual"]  # as in the published Figs 12/13
    p = res[f"{short}.gee_md.expert_vs_novice.p"]
    with plt.rc_context(BAR_STYLE):
        fig, ax = plt.subplots(figsize=(20, 9.5))
        bw = 0.35
        for i, grp in enumerate(groups):
            sub = nav if grp == "Combined" else nav[nav.controller == grp]
            st = sub.groupby("expertise")[col].agg(["mean", "std"])
            xe, xn = i - bw / 2, i + bw / 2
            ax.bar(xe, st.loc["Expert", "mean"], width=bw, yerr=st.loc["Expert", "std"], color=EXPERT_BLUE,
                   error_kw={"lw": 2.0, "ecolor": "black"})
            ax.bar(xn, st.loc["Novice", "mean"], width=bw, yerr=st.loc["Novice", "std"], color=NOVICE_ORANGE,
                   error_kw={"lw": 2.0, "ecolor": "black"})
            ax.text(xe, -y_max * 0.06, "Expert", ha="center", va="top", fontsize=20)
            ax.text(xn, -y_max * 0.06, "Novice", ha="center", va="top", fontsize=20)
            if grp == "Combined":  # the one comparison tested (adjusted GEE)
                top = (st["mean"] + st["std"]).max() + bracket_h
                ax.plot([xe, xe, xn, xn], [top, top + bracket_h, top + bracket_h, top], lw=2.0, c="k")
                ax.text(i, top + bracket_h, stars(p), ha="center", va="bottom", fontsize=26)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=26)
        ax.tick_params(axis="x", length=0)
        for lbl in ax.get_xticklabels():
            lbl.set_y(-0.14)
        ax.set_ylabel(ylabel, fontsize=30)
        ax.set_ylim(-y_max * 0.25, y_max)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        fig.tight_layout()
        save(fig, name, 170, tight=False)


expertise_bars("incorrect_catheterizations", "cath", "Incorrect Catheterizations", 2.0, 0.12,
               "Figs_novice_vs_expert_combined_plot_group_catheterisations_bar_combined_figure.png")
expertise_bars("prolapse_events", "prolapse", "Prolapse Events", 3.6, 0.18,
               "Figs_novice_vs_expert_combined_plot_group_prolapse_bar_combined_figure.png")
print(f"05_figures: 9 figures -> output/figures/")
