import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np, json, os

OK = {"EleutherAI": "#000000", "Llama": "#0072B2", "Qwen": "#E69F00", "OLMo": "#009E73"}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25})
OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/function-vectors/run/followup/002-theory-update/results"
os.makedirs(OUT, exist_ok=True)
M = json.load(open(f"{OUT}/derived_master.json"))
MODELS = M["models"]; A = M["analyses"]
mk = {"MHA": "o", "GQA": "s"}
def short(n): return n.replace("-Base", "").replace("GPT-NeoX-20B", "GPT-NeoX")

fam_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=OK[f], markersize=11, label=f) for f in OK]
attn_h = [Line2D([0], [0], marker=mk[a], color="0.3", markerfacecolor="0.3", linestyle="none", markersize=9, label=a) for a in mk]

# ================= Figure 1: effect size vs era =================
fig, ax = plt.subplots(figsize=(8, 4.8))
yrs = [r["year"] for r in MODELS]; base = [r["baseline"] for r in MODELS]
ax.plot(yrs, base, "--", color="0.6", lw=1, zorder=1)
ax.scatter(yrs, base, color="0.6", s=30, marker="x", zorder=2)
for r in MODELS:
    ax.scatter(r["year"], r["mean_zs_peak"], color=OK[r["family"]], marker=mk[r["attn"]],
               s=160, edgecolor="white", lw=1.2, zorder=4)
    ax.plot([r["year"], r["year"]], [r["baseline"], r["mean_zs_peak"]], color=OK[r["family"]], lw=1.2, alpha=0.5, zorder=3)
    dy = -16 if short(r["name"]) in ("Qwen3-8B", "GPT-NeoX") else 8
    ax.annotate(short(r["name"]), (r["year"], r["mean_zs_peak"]), textcoords="offset points",
                xytext=(6, dy), fontsize=9.5, color=OK[r["family"]])
# trend line (all rungs), flat -> "holds"
s = A["effect_size_vs_era"]["fv_vs_year_all"]["slope"];
import numpy as np
xs = np.array(sorted(set(yrs)), float)
# fit intercept via least squares on the reported slope's regression (recompute cleanly)
yy = np.array([r["mean_zs_peak"] for r in MODELS]); xx = np.array(yrs, float)
b = yy.mean() - s * xx.mean()
ax.plot(xs, s * xs + b, color="0.4", lw=1.4, ls="-", zorder=2)
ax.annotate(f"trend +{s:.1f}%/yr (R²={A['effect_size_vs_era']['fv_vs_year_all']['r2']:.02f}: no monotone era trend)",
            (2021.0, 30), fontsize=9, color="0.35")
ax.set_xlabel("Model release year"); ax.set_ylabel("Mean top-1 zero-shot FV accuracy (%)")
ax.set_xticks(range(2021, 2026)); ax.set_ylim(-3, 100)
base_h = [Line2D([0], [0], marker="x", color="0.6", linestyle="--", markersize=7, label="no-FV baseline")]
ax.legend(handles=fam_h + attn_h + base_h, fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
ax.set_title("Effect size vs era: FV accuracy holds high (45–92%) 2021→2025", fontsize=12)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig1_effect_size_vs_era.{ext}", facecolor="white")
plt.close(fig)

# ================= Figure 2: peak layer vs depth FIT (headline) =================
fig, ax = plt.subplots(figsize=(7.6, 5.2))
Ls = np.array([r["L"] for r in MODELS], float)
fitM = A["peak_layer_fit_MHA"]; fitAll = A["peak_layer_fit_all"]
xg = np.linspace(24, 46, 50)
# MHA through-origin fit line (alpha ~ 1/3)
aM = fitM["alpha_through_origin"]
ax.plot(xg, aM * xg, color="0.3", lw=2.0, zorder=2,
        label=f"MHA fit: peak≈{aM:.2f}·L  (≈L/3)")
ax.plot(xg, xg / 3.0, color="0.6", lw=1.2, ls=":", zorder=1, label="paper canonical L/3")
for r in MODELS:
    ax.scatter(r["L"], r["peak_layer_eff"], color=OK[r["family"]], marker=mk[r["attn"]],
               s=170, edgecolor="white", lw=1.2, zorder=4)
    dx, dy = (6, 6)
    if short(r["name"]) == "GPT-J-6B": dy = -16
    if short(r["name"]) == "OLMo-2-7B": dy = -16
    ax.annotate(short(r["name"]), (r["L"], r["peak_layer_eff"]), textcoords="offset points",
                xytext=(dx, dy), fontsize=9, color=OK[r["family"]])
ax.set_xlabel("Model depth  L  (number of layers)")
ax.set_ylabel("Peak FV edit layer (effective)")
ax.set_xlim(25, 46); ax.set_ylim(0, 24)
ax.set_title("Peak edit layer vs depth: MHA tracks ≈L/3; GQA (Qwen) drifts to ~0.6·L", fontsize=11)
leg1 = ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=fam_h + attn_h, fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
txt = (f"MHA α={fitM['alpha_through_origin']:.2f} (peak_frac {A['peak_frac_by_arch']['MHA_mean']:.2f}±{A['peak_frac_by_arch']['MHA_std']:.2f})\n"
       f"GQA peak_frac {A['peak_frac_by_arch']['GQA_mean']:.2f}±{A['peak_frac_by_arch']['GQA_std']:.2f}\n"
       f"all-rung single slope collapses:\n  α={fitAll['alpha']:.02f}, R²={fitAll['R2']:.02f}")
ax.annotate(txt, (0.98, 0.30), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.2, color="0.25", bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig2_peak_layer_vs_depth_fit.{ext}", facecolor="white")
plt.close(fig)

# ================= Figure 3: concentration vs family/architecture =================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
conc_models = [r for r in MODELS if r["concentration"] is not None]
order = sorted(conc_models, key=lambda r: (r["attn"], -r["concentration"]))
names = [short(r["name"]) for r in order]
cols = [OK[r["family"]] for r in order]
# left: concentration (frac top-K in first 2/3), grouped by arch
a1.bar(range(len(order)), [r["concentration"] for r in order], color=cols, edgecolor="white")
a1.axhline(0.667, color="0.3", ls=":", lw=1)
a1.annotate("first ⅔ threshold", (len(order)-0.5, 0.68), fontsize=7.5, color="0.4", ha="right")
a1.axhline(A["concentration_by_arch"]["MHA_concentration_mean"], color="0.5", ls="--", lw=0.8)
a1.set_ylim(0, 1.08); a1.set_xticks(range(len(order))); a1.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
a1.set_ylabel("Top-AIE heads in first ⅔ of layers")
a1.set_title(f"Concentration: MHA {A['concentration_by_arch']['MHA_concentration_mean']:.2f} vs GQA {A['concentration_by_arch']['GQA_concentration_mean']:.2f}", fontsize=10.5)
# right: mean layer fraction of top-K heads, MHA vs GQA
lf = [r for r in conc_models if r["mean_layer_frac"] is not None]
lf = sorted(lf, key=lambda r: (r["attn"], r["mean_layer_frac"]))
a2.bar(range(len(lf)), [r["mean_layer_frac"] for r in lf], color=[OK[r["family"]] for r in lf], edgecolor="white")
a2.axhline(0.333, color="0.6", ls=":", lw=1); a2.annotate("L/3", (len(lf)-0.5, 0.35), fontsize=8, color="0.5", ha="right")
a2.set_ylim(0, 0.8); a2.set_xticks(range(len(lf))); a2.set_xticklabels([short(r["name"]) for r in lf], rotation=35, ha="right", fontsize=8.5)
a2.set_ylabel("Mean layer-fraction of top-K AIE heads")
a2.set_title(f"Localization depth: MHA {A['concentration_by_arch']['MHA_mean_layer_frac']:.2f} vs GQA {A['concentration_by_arch']['GQA_mean_layer_frac']:.2f}", fontsize=10.5)
arch_h = [Line2D([0],[0],marker="s",color="w",markerfacecolor=OK[f],markersize=10,label=f) for f in OK if f != "EleutherAI"] + \
         [Line2D([0],[0],marker="s",color="w",markerfacecolor=OK["EleutherAI"],markersize=10,label="EleutherAI")]
fig.legend(handles=arch_h, fontsize=8, ncol=4, loc="lower center", framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Head localization vs architecture: signature preserved; GQA sits deeper & (Qwen2.5) more diffuse", fontsize=11.5, y=1.0)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig3_concentration_vs_family.{ext}", facecolor="white")
plt.close(fig)
print("wrote figures to", OUT)
print([f for f in sorted(os.listdir(OUT)) if f.endswith(".png")])
