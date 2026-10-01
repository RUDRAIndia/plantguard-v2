"""
Day 11: Generate report figures and tables from artifacts/results.json.
Run from repo root with the .venv activated:
    python scripts/generate_report_figures.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "artifacts" / "results.json"
OUT_DIR = REPO / "artifacts" / "figures" / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS, "r", encoding="utf-8") as f:
    data = json.load(f)

# Colourblind-safe palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("Loaded results.json. Keys:", list(data.keys()))
# --- Figure 1: four-model comparison bar chart ---
ranking = data["model_selection"]["ranking"]
names = [m["model_name"] for m in ranking]
f1_scores = [m["val_macro_f1"] for m in ranking]
params = [m["param_count"] for m in ranking]
sizes_mb = [m["checkpoint_file_size_bytes"] / (1024 * 1024) for m in ranking]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, f1_scores, color=COLORS[: len(names)])
for bar, p, s in zip(bars, params, sizes_mb):
    height = bar.get_height()
    ax.annotate(
        f"{height:.4f}\n{p:,} params\n{s:.1f} MB",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("Validation macro-F1")
#ax.set_title("Four Mobile-Class Architectures: Validation macro-F1")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_model_comparison.png", dpi=300)
plt.close()
print("Wrote fig1_model_comparison.png")
# --- Figure 2: headline PlantVillage vs PlantDoc comparison ---
pv_f1 = data["test_evaluation"]["macro_f1"]
pd_f1 = data["external_evaluation_plantdoc"]["overall"]["macro_f1"]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(
    ["PlantVillage\n(lab test set)", "PlantDoc\n(real field photos)"],
    [pv_f1, pd_f1],
    color=[COLORS[0], COLORS[3]],
)
for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
ax.set_ylabel("macro-F1")
ax.set_title("The Lab-to-Field Generalisation Gap (MobileNetV3Large)")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_headline_gap.png", dpi=300)
plt.close()
print("Wrote fig2_headline_gap.png")
# --- Figure 3: PlantDoc mapping confidence tier breakdown ---
tiers = data["external_evaluation_plantdoc"]["mapping_confidence_tier_breakdown"]
tier_names = list(tiers.keys())
tier_f1 = [tiers[t]["macro_f1"] for t in tier_names]
tier_counts = [tiers[t]["num_images"] for t in tier_names]

fig, ax1 = plt.subplots(figsize=(7, 5))
x = range(len(tier_names))
bars = ax1.bar(x, tier_f1, color=COLORS[: len(tier_names)])
ax1.set_xticks(list(x))
ax1.set_xticklabels([t.capitalize() for t in tier_names])
ax1.set_ylabel("macro-F1")
ax1.set_title("PlantDoc Mapping Confidence Tiers")
for bar, count in zip(bars, tier_counts):
    height = bar.get_height()
    ax1.annotate(
        f"{height:.4f}\n(n={count})",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_mapping_tiers.png", dpi=300)
plt.close()
print("Wrote fig3_mapping_tiers.png")
# --- Figure 4: OOD threshold sweep with two y-series, deployed threshold marked ---
rows = data["ood_rejection"]["ood_threshold_sweep"]["rows"]
thresholds = [r["threshold"] for r in rows]
confident_wrong_pct = [r["plantdoc"]["pct_confident_wrong"] * 100 for r in rows]
val_rejected_pct = [r["validation"]["rejection_rate"] * 100 for r in rows]
deployed_threshold = data["ood_rejection"]["chosen_threshold"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, confident_wrong_pct, marker="o", color=COLORS[3],
        label="PlantDoc confident-wrong %")
ax.plot(thresholds, val_rejected_pct, marker="s", color=COLORS[0],
        label="Validation false-rejection %")
ax.axvline(deployed_threshold, color="gray", linestyle="--",
           label=f"Deployed threshold ({deployed_threshold})")
ax.set_xlabel("OOD confidence threshold")
ax.set_ylabel("Percentage")
ax.set_title("OOD Threshold Sweep: Safety vs. Usability Trade-off")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_threshold_sweep.png", dpi=300)
plt.close()
print("Wrote fig4_threshold_sweep.png")
# --- Figure 5: robustness heatmap (corruption type x severity) ---
import numpy as np

corruption_types = ["blur", "gaussian_noise", "brightness_up", "brightness_down", "rotation", "jpeg_compression"]
robustness = data["robustness"]

# collect all severity keys present, sorted
severities = sorted(
    {sev for ctype in corruption_types for sev in robustness[ctype].keys()},
    key=lambda s: float(s)
)

heatmap_data = np.zeros((len(corruption_types), len(severities)))
for i, ctype in enumerate(corruption_types):
    for j, sev in enumerate(severities):
        if sev in robustness[ctype]:
            heatmap_data[i, j] = robustness[ctype][sev]["macro_f1"]
        else:
            heatmap_data[i, j] = np.nan

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(severities)))
ax.set_xticklabels(severities)
ax.set_yticks(range(len(corruption_types)))
ax.set_yticklabels([c.replace("_", " ").title() for c in corruption_types])
ax.set_xlabel("Severity level")
ax.set_title("Robustness Under Corruption (macro-F1)")
for i in range(len(corruption_types)):
    for j in range(len(severities)):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                     color="black" if val > 0.5 else "white")
plt.colorbar(im, ax=ax, label="macro-F1")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_robustness_heatmap.png", dpi=300)
plt.close()
print("Wrote fig5_robustness_heatmap.png")
# --- Figure 6: top-10 confused pairs, as a clean table image ---
pairs = data["test_evaluation"]["top_confused_pairs"][:10]

fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")

col_labels = ["True class", "Predicted as", "Count", "Rate of true class"]
table_data = []
for p in pairs:
    true_short = p["true_class"].replace("_", " ")
    pred_short = p["predicted_class"].replace("_", " ")
    table_data.append([true_short, pred_short, str(p["count"]), f"{p['rate_of_true_class']*100:.1f}%"])

table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor(COLORS[0])
    table[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Top-10 Confused Class Pairs (PlantVillage Test Set)", pad=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_confused_pairs.png", dpi=300)
plt.close()
print("Wrote fig6_confused_pairs.png")
# --- report_tables.md: every table as markdown, pasteable into the report ---
lines = []
lines.append("# PlantGuard Report Tables\n")
lines.append(f"_Generated from results.json (commit `{data['git_commit_hash']}`)_\n")

lines.append("## Table 1: Model comparison\n")
lines.append("| Model | Val macro-F1 | Params | Checkpoint size |")
lines.append("|---|---|---|---|")
for m in ranking:
    size_mb = m["checkpoint_file_size_bytes"] / (1024 * 1024)
    lines.append(f"| {m['model_name']} | {m['val_macro_f1']:.4f} | {m['param_count']:,} | {size_mb:.1f} MB |")
lines.append("")

lines.append("## Table 2: PlantVillage test set vs PlantDoc field set\n")
lines.append("| Metric | PlantVillage (test) | PlantDoc (field) |")
lines.append("|---|---|---|")
lines.append(f"| Accuracy | {data['test_evaluation']['accuracy']:.4f} | {data['external_evaluation_plantdoc']['overall']['accuracy']:.4f} |")
lines.append(f"| Macro-F1 | {data['test_evaluation']['macro_f1']:.4f} | {data['external_evaluation_plantdoc']['overall']['macro_f1']:.4f} |")
lines.append("")

lines.append("## Table 3: PlantDoc mapping confidence tiers\n")
lines.append("| Tier | macro-F1 | Num images |")
lines.append("|---|---|---|")
for t in tier_names:
    lines.append(f"| {t.capitalize()} | {tiers[t]['macro_f1']:.4f} | {tiers[t]['num_images']} |")
lines.append("")

lines.append("## Table 4: OOD threshold sweep (selected rows)\n")
lines.append("| Threshold | PlantDoc confident-wrong % | Val false-rejection % |")
lines.append("|---|---|---|")
for r in rows:
    if r["threshold"] in (0.0, 0.7, 0.9, 0.95, 0.98, 1.0):
        lines.append(f"| {r['threshold']:.2f} | {r['plantdoc']['pct_confident_wrong']*100:.1f}% | {r['validation']['rejection_rate']*100:.1f}% |")
lines.append("")

lines.append("## Table 5: Robustness under corruption (macro-F1)\n")
lines.append("| Corruption | " + " | ".join(f"Severity {s}" for s in severities) + " |")
lines.append("|---|" + "---|" * len(severities))
for ctype in corruption_types:
    row_vals = []
    for sev in severities:
        if sev in robustness[ctype]:
            row_vals.append(f"{robustness[ctype][sev]['macro_f1']:.4f}")
        else:
            row_vals.append("-")
    lines.append(f"| {ctype.replace('_', ' ').title()} | " + " | ".join(row_vals) + " |")
lines.append("")

lines.append("## Table 6: Top-10 confused pairs\n")
lines.append("| True class | Predicted as | Count | Rate of true class |")
lines.append("|---|---|---|---|")
for p in pairs:
    lines.append(f"| {p['true_class']} | {p['predicted_class']} | {p['count']} | {p['rate_of_true_class']*100:.1f}% |")
lines.append("")

lines.append("## Calibration\n")
cal = data["calibration"]
lines.append(f"- Temperature: {cal['temperature']:.2f}")
lines.append(f"- Test ECE before: {cal['test_ece_before']:.4f}")
lines.append(f"- Test ECE after: {cal['test_ece_after']:.4f}")
lines.append("")

lines.append("## OOD rejection at deployed threshold (0.98)\n")
ood = data["ood_rejection"]
lines.append(f"- Deployed threshold: {ood['chosen_threshold']}")
lines.append(f"- PlantDoc rejection rate: {ood['plantdoc_at_chosen_threshold'].get('rejection_rate', 'see JSON')}")
lines.append("")

with open(REPO / "artifacts" / "report_tables.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Wrote artifacts/report_tables.md")