import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np, json, os

OK = {"EleutherAI":"#000000","Llama":"#0072B2","Qwen":"#E69F00","OLMo":"#009E73"}
plt.rcParams.update({
    "figure.dpi":150,"savefig.dpi":150,"savefig.bbox":"tight",
    "font.size":12,"axes.titlesize":13,"axes.labelsize":12,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25})
OUT="/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/function-vectors/run/followup/001-living-update/results"
os.makedirs(OUT,exist_ok=True)

# (model, year, family, attn, mean_peak_ZS_FV%, no-FV baseline%, peak_frac(effective-task mean), note)
D=[
 ("GPT-J-6B",2021,"EleutherAI","MHA",65.9,3.2,0.361,"anchor (reused, fp16)"),
 ("GPT-NeoX-20B",2022,"EleutherAI","MHA",57.1,2.1,0.273,"antonym-only (reused)"),
 ("Llama-2-7B",2023,"Llama","MHA",92.2,8.7,0.260,""),
 ("Qwen2.5-7B",2024,"Qwen","GQA",74.6,1.8,0.643,""),
 ("Qwen3-8B",2025,"Qwen","GQA",45.0,2.3,0.545,"country-capital FV failed"),
 ("OLMo-2-7B",2025,"OLMo","MHA",89.3,2.4,0.365,""),
]
mk={"MHA":"o","GQA":"s"}

# ---------- Figure 1: FV accuracy over time ----------
fig,ax=plt.subplots(figsize=(8,4.8))
# baseline faint line
yrs=[d[1] for d in D]; base=[d[5] for d in D]
ax.plot(yrs,base,"--",color="0.6",lw=1,zorder=1)
ax.scatter(yrs,base,color="0.6",s=30,marker="x",zorder=2,label="no-FV baseline (0-shot)")
for name,yr,fam,attn,fv,bl,pf,note in D:
    ax.scatter(yr,fv,color=OK[fam],marker=mk[attn],s=160,edgecolor="white",lw=1.2,zorder=4)
    ax.plot([yr,yr],[bl,fv],color=OK[fam],lw=1.2,alpha=0.5,zorder=3)
    dy = -16 if name in ("Qwen3-8B","GPT-NeoX-20B") else 8
    ax.annotate(name,(yr,fv),textcoords="offset points",xytext=(6,dy),fontsize=9.5,color=OK[fam])
ax.axhline(55.0,color="0.3",lw=0.8,ls=":",zorder=1)
ax.annotate("GPT-J paper canonical 57.5 / replicated 55.0% (6-task, @L9)",(2021,55.0),
            textcoords="offset points",xytext=(2,-14),fontsize=8,color="0.35")
ax.set_xlabel("Model release year"); ax.set_ylabel("Mean top-1 zero-shot FV accuracy (%)")
ax.set_xticks(range(2021,2026)); ax.set_ylim(-3,100)
fam_h=[Line2D([0],[0],marker="o",color="w",markerfacecolor=OK[f],markersize=11,label=f) for f in OK]
attn_h=[Line2D([0],[0],marker=mk[a],color="0.3",markerfacecolor="0.3",linestyle="none",markersize=9,label=a) for a in mk]
base_h=[Line2D([0],[0],marker="x",color="0.6",linestyle="--",markersize=7,label="no-FV baseline")]
ax.legend(handles=fam_h+attn_h+base_h,fontsize=8,ncol=2,loc="lower right",framealpha=0.9)
ax.set_title("Function-vector zero-shot accuracy at peak edit layer, 2021→2025",fontsize=12)
for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/fv_over_time.{ext}",facecolor="white")
fig.savefig(f"{OUT}/fig1_fv_over_time.png",facecolor="white")
plt.close(fig)

# ---------- Figure 2: peak edit-layer as fraction of depth ----------
fig,ax=plt.subplots(figsize=(8,4.4))
ax.axhspan(0.20,0.45,color="0.85",alpha=0.5,zorder=0)
ax.annotate("paper 'early-mid' band",(2021.05,0.43),fontsize=8.5,color="0.4")
for name,yr,fam,attn,fv,bl,pf,note in D:
    ax.scatter(yr,pf,color=OK[fam],marker=mk[attn],s=160,edgecolor="white",lw=1.2,zorder=4)
    dy = 8 if name!="Qwen2.5-7B" else 6
    ax.annotate(name,(yr,pf),textcoords="offset points",xytext=(6,dy),fontsize=9.5,color=OK[fam])
ax.set_xlabel("Model release year"); ax.set_ylabel("Peak FV edit layer (fraction of depth)")
ax.set_xticks(range(2021,2026)); ax.set_ylim(0,0.8)
ax.legend(handles=fam_h+attn_h,fontsize=8,ncol=2,loc="upper left",framealpha=0.9)
ax.set_title("Where the FV causally triggers: peak layer drifts later for Qwen (GQA)",fontsize=11.5)
for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/fv_peak_layer_vs_depth.{ext}",facecolor="white")
fig.savefig(f"{OUT}/fig2_fv_peak_layer_vs_depth.png",facecolor="white")
plt.close(fig)

# ---------- Figure 3: head localization (concentration + induction overlap) ----------
HL={"Llama-2-7B":(0.95,0.0479,0.0255,"Llama"),
    "Qwen2.5-7B":(0.611,0.0537,0.0438,"Qwen"),
    "Qwen3-8B":(1.0,0.0752,0.0385,"Qwen"),
    "OLMo-2-7B":(0.95,0.0187,0.0252,"OLMo"),
    "GPT-J-6B":(0.9,0.1821,0.0436,"EleutherAI")}  # GPT-J: 9/10 top-AIE in early-mid; pms top10 mean vs all (head_localization.txt)
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.5,4.2))
names=list(HL.keys())
conc=[HL[n][0] for n in names]; cols=[OK[HL[n][3]] for n in names]
a1.bar(range(len(names)),conc,color=cols,edgecolor="white")
a1.axhline(0.667,color="0.3",ls=":",lw=1); a1.set_ylim(0,1.05)
a1.set_xticks(range(len(names))); a1.set_xticklabels(names,rotation=35,ha="right",fontsize=8.5)
a1.set_ylabel("Top-AIE heads in first ⅔ of layers"); a1.set_title("Mid-layer concentration (C5)",fontsize=11)
x=np.arange(len(names)); w=0.38
a2.bar(x-w/2,[HL[n][1] for n in names],w,color=cols,label="top-K AIE heads",edgecolor="white")
a2.bar(x+w/2,[HL[n][2] for n in names],w,color="0.7",label="all heads (mean)",edgecolor="white")
a2.set_xticks(x); a2.set_xticklabels(names,rotation=35,ha="right",fontsize=8.5)
a2.set_ylabel("Prefix-matching (induction) score"); a2.set_title("Induction-head overlap (C13)",fontsize=11)
a2.legend(fontsize=8.5)
fig.suptitle("Head-localization signatures persist under the MHA→GQA shift",fontsize=12,y=1.02)
for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/fv_head_concentration.{ext}",facecolor="white")
fig.savefig(f"{OUT}/fig3_fv_head_concentration.png",facecolor="white")
plt.close(fig)
print("figures written to",OUT)
print(os.listdir(OUT))
EOF_MARKER=1
