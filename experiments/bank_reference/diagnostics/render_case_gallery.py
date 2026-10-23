"""Render archived PM case cards; no model, training, or new scientific arms."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


BLUE = "#2563a6"
ORANGE = "#cf552c"
GREEN = "#237a63"
GRAY = "#718096"
COLORS = [BLUE, ORANGE, GREEN]
ORDER = {"EXACT": 0, "NEWPATH": 1, "CHANGE": 2}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def parent_key(row):
    return ORDER[row["condition"]], row["source_index"], row["original_t"]


def eligible(rows):
    noise = [r for r in rows if r["method"] == "K64/PaFu"
             and r["positive_task_regret"] and r["oracle_worsens_signal"]]
    success = [r for r in rows if not r["positive_task_regret"]
               and r["chosen_minus_zero"]["total"] < -1e-8
               and r["chosen_minus_zero"]["signal"] < -1e-8]
    return noise, success


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#243746",
                         "text.color": "#182c3a", "savefig.facecolor": "white"})


def identity(row):
    return f"{row['condition']} / source {row['source_index']} / {row['method']} / t={row['original_t']}"


def draw_bank(row, slug, title, takeaway, denominator, success=False):
    fig = plt.figure(figsize=(13.3, 9.1), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, .11, 1, .89))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.5, 1.35])
    fig.suptitle(f"{title}\n{identity(row)}", fontsize=16, fontweight="bold")
    timeline = fig.add_subplot(grid[0, :])
    chosen, oracle = row["chosen_drop"], row["oracle_drop"]
    t = np.asarray(row["candidate_original_times"])
    prob = np.asarray(row["teacher"]["probability"])
    timeline.scatter(t, np.zeros(len(t)), color="#c9d4dc", s=25, label="K+1 candidates")
    hits = np.flatnonzero(prob > 0)
    for n, i in enumerate(hits):
        timeline.scatter(t[i], 0, s=45 + 180 * prob[i], color=BLUE, zorder=3)
        timeline.annotate(f"t{t[i]} / key {row['candidate_keys'][i]}\nattn {prob[i]:.2f}",
                          (t[i], 0), xytext=(.05 + .11 * n, .83 - .22 * (n % 2)),
                          textcoords="axes fraction", fontsize=8, ha="center",
                          arrowprops={"arrowstyle": "-", "color": BLUE, "lw": .7})
    timeline.scatter([t[chosen]], [-.22], marker="x", color=ORANGE, s=70,
                     label=f"PaFu drops t{t[chosen]} / key {row['candidate_keys'][chosen]}")
    timeline.scatter([t[oracle]], [-.44], marker="x", color=GREEN, s=70,
                     label=f"Oracle drops t{t[oracle]} / key {row['candidate_keys'][oracle]}")
    timeline.axvline(row["original_t"], color=GRAY, ls="--", lw=1)
    timeline.set(xlim=(-8, row["original_t"] + 8), ylim=(-.64, 1.1), yticks=[],
                 xlabel="Original observed time (key IDs are categorical, not spatial coordinates)")
    timeline.set_title("A. Observed candidate history and archived future-attention labels", loc="left", fontsize=11)
    timeline.legend(loc="upper right", fontsize=8, frameon=False)
    for spine in ["left", "top", "right"]:
        timeline.spines[spine].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    values = [row["candidate_parts"][chosen]["total"],
              row["candidate_parts"][oracle]["total"], row["zero_memory_parts"]["total"]]
    bars = ax.bar(["PaFu memory", "Local oracle", "Zero residual\n(reference)"], values,
                  color=[ORANGE, GREEN, GRAY], width=.62)
    ax.bar_label(bars, labels=[f"{v:.6f}" for v in values], padding=4, fontsize=10)
    ax.set(ylim=(0, max(values) * 1.25), ylabel="Six-query masked MPE (lower is better)")
    ax.set_title("B. Same K / same reader / same future objective", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=.15)
    c, o = [row["candidate_scored_coverage"][i] for i in [chosen, oracle]]
    ax.text(.02, .96, f"Scored history coverage: {c:.1%} -> {o:.1%}",
            transform=ax.transAxes, fontsize=9, va="top")

    ax = fig.add_subplot(grid[1, 1])
    if success:
        delta = {k: -v for k, v in row["chosen_minus_zero"].items()}
        subtitle = "C. Benefit of using memory: zero minus PaFu"
    else:
        delta = row["chosen_minus_oracle"]
        subtitle = "C. Benefit of oracle deletion: PaFu minus oracle"
    keys = ["signal", "noise", "cross", "total"]
    vals = [delta[k] for k in keys]
    bars = ax.bar(keys, vals, color=[BLUE, ORANGE, GRAY, GREEN], width=.62)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:+.6f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#526775", lw=.8)
    lo, hi = min(0, min(vals)), max(vals)
    span = hi - lo or 1
    ax.set(ylim=(lo - .22 * span, hi + .25 * span), ylabel="Positive = lower error; negative = cost")
    ax.set_title(subtitle, loc="left", fontsize=11)
    ax.grid(axis="y", alpha=.15)

    ax = fig.add_subplot(grid[2, 0])
    lookup = {q["drop"]: q["queries"] for q in row["query_detail"]}
    for drop, label, color, marker in [(chosen, "PaFu", ORANGE, "o"), (oracle, "Oracle", GREEN, "s")]:
        q = lookup[drop]
        ax.plot(range(6), [z["parts"]["total"] for z in q], marker=marker,
                color=color, label=label, lw=1.6, ms=5)
    q = lookup[chosen]
    ax.set_xticks(range(6), [f"t{z['original_t']}\nk{z['query_key']}" for z in q], fontsize=8)
    ax.set(ylabel="Per-query masked MPE", xlabel="Actual future query time / oracle key")
    ax.set_title("D. Which future queries change?", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.15)

    ax = fig.add_subplot(grid[2, 1])
    table_data = []
    for j, q in enumerate(lookup[chosen]):
        o = lookup[oracle][j]
        table_data.append([f"{q['original_t']} / {q['query_key']}",
                           f"{q['covered_scored_dimensions']}/{q['scored_dimensions']}",
                           f"{o['covered_scored_dimensions']}/{o['scored_dimensions']}"])
    ax.axis("off")
    ax.set_title("E. Covered scored coordinates / total scored", loc="left", fontsize=11)
    table = ax.table(cellText=table_data, colLabels=["time / key", "PaFu", "Oracle"],
                     loc="upper center", cellLoc="center", colLoc="center", bbox=[0, .1, 1, .82])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ea")
        if r == 0:
            cell.set_facecolor("#edf3f7")
    fig.text(.5, .012, f"{takeaway}\n{denominator}\n"
                  "Archived residual accounting, not RGB. Signal is pre-injection slow-model residual, not physical ground truth.",
                  fontsize=9, ha="center", va="bottom")
    return fig


def draw_a2(conclusion):
    fig, axs = plt.subplots(1, 2, figsize=(13.3, 7.1), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, .13, 1, .87))
    fig.suptitle("04 | A2 trades signal fidelity for denoising\nEXACT object coordinates: existing aggregate, not a reconstructed scene", fontsize=16, fontweight="bold")
    vals = [0.010935638, -0.004822307, 0.000780184]
    total = sum(vals)
    bars = axs[0].bar(["Signal cost", "Noise saving", "Cross cost", "Net coordinate cost"],
                       vals + [total], color=[BLUE, GREEN, GRAY, ORANGE], width=.65)
    axs[0].bar_label(bars, labels=[f"{v:+.6f}" for v in vals + [total]], padding=4)
    axs[0].axhline(0, color=GRAY, lw=.8)
    axs[0].set(ylim=(-.008, .015), ylabel="A2 minus A1 coordinate MPE (positive is worse)")
    axs[0].tick_params(axis="x", labelsize=8)
    axs[0].set_title("A. The localized cost is not noise amplification", loc="left", fontsize=11)
    axs[0].grid(axis="y", alpha=.15)
    axs[1].barh(["Other observed dimensions", "Object coordinates", "Grid"],
                 [(1 - .778094 - .033267) * 100, .033267 * 100, .778094 * 100],
                 color=[GRAY, ORANGE, BLUE], height=.55)
    axs[1].set(xlabel="Share of the EXACT observed-loss denominator (%)", xlim=(0, 100))
    for i, v in enumerate([(1 - .778094 - .033267) * 100, .033267 * 100, .778094 * 100]):
        axs[1].text(v + 1, i, f"{v:.2f}%", va="center")
    axs[1].set_title("B. A rare coordinate cost can be hidden in the average", loc="left", fontsize=11)
    rec = conclusion["recovering_only_observed_exact_coordinate_loss"]
    fig.text(.5, .015, "CHANGE coordinate error improves; a general object-memory failure was not established.\n"
                  f"Bookkeeping: fixing only this observed cost would change combined retention MPE by {rec['retention_mpe_recovery']:.7f} ({rec['retention_relative_to_A2_percent']:.4f}%).\n"
                  "Not a tested policy, attainable guarantee, or bound on all methods. Existing negative main-line decision is preserved.",
                  fontsize=10, ha="center", va="bottom")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((out / "selection_protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert digest(args.source / name) == expected, name
    rows = json.loads((args.source / "parents.json").read_text())
    old = json.loads((args.source / "previous_top_case.json").read_text())
    a2 = json.loads((args.source / "prior_a2_conclusion.json").read_text())
    noise, successes = eligible(rows)
    selected_noise = sorted(noise, key=parent_key)[0]
    selected_success = sorted([r for r in successes if r["method"] == "K64/PaFu"], key=parent_key)[0]
    assert any(r == old for r in rows)
    by_method = {}
    for method in sorted({r["method"] for r in rows}):
        members = [r for r in rows if r["method"] == method]
        by_method[method] = {"parents": len(members),
                             "positive_local_regret": sum(r["positive_task_regret"] for r in members),
                             "oracle_worsens_signal_with_positive_regret": sum(r["oracle_worsens_signal"] and r["positive_task_regret"] for r in members),
                             "baseline_beats_zero_total": sum(r["chosen_minus_zero"]["total"] < -1e-8 for r in members),
                             "positive_control_rule": sum(r in successes for r in members)}
    population = {"parents": len(rows), "unique_A_sources": 8,
                  "conditions": dict(Counter(r["condition"] for r in rows)),
                  "positive_local_regret": sum(r["positive_task_regret"] for r in rows),
                  "oracle_worsens_signal_with_positive_regret": sum(r["oracle_worsens_signal"] and r["positive_task_regret"] for r in rows),
                  "positive_control_rule": len(successes), "by_method": by_method,
                  "not_independent": "Repeated anchors/policies share source trajectories; CHANGE B sources overlap."}
    dump(out / "population_counts.json", population)
    dump(out / "selected_cases.json", {"harmful_read": old, "noise_tradeoff": selected_noise,
                                       "positive_control": selected_success, "a2": a2})
    style()
    cases = [
        ("01_harmful_read", draw_bank(old, "01", "01 | Attention selects a harmful old value",
          "Deleting the high-attention entry helps mainly through signal; zero residual is still better than oracle.",
          "Previously selected case reused; 129/864 parents have positive local regret, 81/129 improve signal.")),
        ("02_noise_tradeoff", draw_bank(selected_noise, "02", "02 | Lower oracle loss can sacrifice signal",
          "Coverage is unchanged: deleting one same-key record lowers realized noise but worsens signal.",
          "Deterministic first qualifying K64/PaFu case: 24/96 qualify; all methods: 48/129 positive-regret parents worsen signal.")),
        ("03_positive_control", draw_bank(selected_success, "03", "03 | A positive control: memory is useful here",
          "PaFu and local oracle tie; memory improves signal and total loss versus zero, despite a noise cost.",
          "Deterministic first qualifying K64/PaFu case: 3/96 qualify; all methods: 25/864. This is one snapshot, not online superiority.", True)),
        ("04_a2_tradeoff", draw_a2(a2)),
    ]
    outputs = []
    with PdfPages(out / "pm_case_cards.pdf") as pdf:
        for name, fig in cases:
            fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            outputs += [f"{name}.png", f"{name}.pdf"]
    outputs += ["pm_case_cards.pdf", "selected_cases.json", "population_counts.json"]
    audit = {"source_sha256": protocol["source_sha256"],
             "renderer_sha256": digest(__file__), "new_model_calls": 0, "new_training_steps": 0,
             "new_scenarios": 0, "selected_ids": {k: identity(v) for k, v in {
                 "harmful_read": old, "noise_tradeoff": selected_noise, "positive_control": selected_success}.items()},
             "outputs": {name: {"sha256": digest(out / name), "bytes": (out / name).stat().st_size} for name in outputs}}
    dump(out / "render_manifest.json", audit)
    print(json.dumps({"status": "rendered_existing_results", "selected_ids": audit["selected_ids"],
                      "population": {k: v for k, v in population.items() if k != "by_method"}}, indent=2))


if __name__ == "__main__":
    main()
