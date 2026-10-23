"""Plot saved BR-0.1 results only; performs no model inference or fitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / "results/br-0.1-dev-seed17"
summary = json.loads((root / "summary.json").read_text())
ks = [4, 16, 64]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
colors = {"FIFO": "#6b7280", "WR": "#c76d34", "PaFu": "#246b9b"}
for ax, metric, title in zip(axes[:2], ["RETENTION/POST", "CHANGE/POST"], ["Retention (EXACT + NEWPATH)", "CHANGE"]):
    values = summary["metrics"][metric]["mpe"]
    for method, color in colors.items():
        ax.plot(ks, [values[f"K{k}/{method}"] for k in ks], "o-", label=method + " adapter", color=color)
    ax.axhline(values["REF/no_memory"], color="#38413b", ls="--", label="No memory")
    ax.axhline(values["REF/A2"], color="#7c5aa5", ls=":", label="A2 replay reference")
    ax.set(title=title, xlabel="Historical item capacity K", ylabel="Residual MPE (lower is better)")
    ax.set_xscale("log", base=2); ax.set_xticks(ks, ks)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
axes[0].legend(fontsize=8, loc="lower right")
counts = [summary["training"][str(k)]["records"] for k in ks]
axes[2].bar([str(k) for k in ks], counts, color="#246b9b")
for i, count in enumerate(counts):
    axes[2].text(i, count + 1, f"{count}/276", ha="center", fontsize=10)
axes[2].set(title="Actual future-attention coverage", xlabel="Historical item capacity K", ylabel="Informative training parents", ylim=(0, 50))
axes[2].spines[["top", "right"]].set_visible(False)
fig.suptitle("Frozen-residual development diagnostic — not a WR/PaFu paper reproduction", fontsize=12)
fig.savefig(root / "development-summary.png", dpi=180)
svg_path = root / "development-summary.svg"
fig.savefig(svg_path)
# Matplotlib emits trailing spaces in SVG paths; keep generated evidence clean.
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
plt.close(fig)
