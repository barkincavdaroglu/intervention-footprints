#!/usr/bin/env python3
"""Regenerate five companion-post figures from the included saved measurements.

Usage: python notes/later-controls/companion/make_figures.py --output /tmp/paper-b-figures
Requires the repository's requirements-figures.txt. No model execution.
"""
from pathlib import Path
import argparse, hashlib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("/tmp/ablation-resolution-figures"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = Path(__file__).with_name("figure-data.json")
data = json.loads(source.read_text())
records = data["records"]
cells = ["amazon_computers/thesis_gcn", "amazon_computers/residual_gcn", "roman_empire/thesis_gcn", "roman_empire/residual_gcn"]
labels = ["Amazon · narrow GCN", "Amazon · residual GCN", "Roman Empire · narrow GCN", "Roman Empire · residual GCN"]
colors = ["#42698d", "#a58848", "#56806e", "#b16f66"]
color = dict(zip(cells, colors))
bg, ink, grid = "#fafaf7", "#272d2b", "#dedfd8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "normal", "axes.labelsize": 10, "figure.facecolor": bg,
    "axes.facecolor": bg, "text.color": ink, "axes.labelcolor": ink,
    "xtick.color": ink, "ytick.color": ink, "axes.edgecolor": grid,
    "savefig.facecolor": bg, "svg.fonttype": "none", "pdf.fonttype": 42})

def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=grid, alpha=.65, linewidth=.7)

def legend(fig):
    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=l, markersize=6) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(.5, -.012))

def dots(ax, categories, table, keys, transform=lambda x:x, threshold=None, annotate=True):
    """Color identifies cells; short bars show cell medians, diamonds pooled medians."""
    for row, key in enumerate(keys):
        allvals = []
        for ci, cell in enumerate(cells):
            vals = np.array([transform(r[table][key]) for r in records if r["cell"] == cell])
            allvals.extend(vals)
            yc = row + (ci - 1.5) * .10
            jitter = np.linspace(-.033, .033, len(vals))
            ax.scatter(vals, yc + jitter, c=color[cell], s=24, alpha=.72, edgecolors="none", zorder=3)
            med = float(np.median(vals))
            ax.plot([med, med], [yc - .045, yc + .045], color=color[cell], lw=3, zorder=4)
        pooled = transform(float(np.median([r[table][key] for r in records])))
        ax.scatter([pooled], [row + .27], marker="D", c=ink, s=25, zorder=5)
        if annotate:
            text = f"{pooled:.2f}×" if transform is not identity else f"{pooled:+.4f}"
            ax.annotate(text, (pooled, row + .27), xytext=(6, -5), textcoords="offset points", fontsize=9)
    ax.set_yticks(range(len(keys)), categories)
    ax.set_ylim(len(keys)-.5, -.5)
    if threshold is not None: ax.axvline(threshold, color="#777c75", ls=(0,(4,4)), lw=1)
    style(ax)

def identity(x): return x

def cell_dots(ax, table, key, xlim=None):
    for i, cell in enumerate(cells):
        vals = np.array([r[table][key] for r in records if r["cell"]==cell])
        ax.scatter(vals, i + np.linspace(-.12,.12,len(vals)), color=color[cell], s=27, alpha=.72)
        med = float(np.median(vals))
        ax.plot([med, med], [i-.19,i+.19], color=color[cell], lw=3)
        ax.annotate(f"{med:.4f}", (med,i), xytext=(0,-23), textcoords="offset points", fontsize=9, ha="center")
    ax.set_yticks(range(4),labels)
    ax.set_ylim(3.5,-.5)
    if xlim: ax.set_xlim(*xlim)
    style(ax)

outputs=[]
def save(fig, name):
    for extension in ("png","svg"):
        path=args.output/(name+"."+extension)
        fig.savefig(path,dpi=190,bbox_inches="tight")
        outputs.append({"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)

# Figure 1: one transparently chosen specimen plus all-checkpoint evidence.
fig,axes=plt.subplots(1,2,figsize=(12.6,4.8),gridspec_kw={"width_ratios":[1,1.25]})
e=data["example"]
x=np.array(e["signed_exact"]);y=np.array(e["signed_approximate"])
scale=np.max(np.abs(x)); x=x/scale;y=y/scale
axes[0].scatter(x,y,s=10,alpha=.30,color=color[e["cell"]],edgecolors="none",rasterized=True)
lower=min(float(x.min()),float(y.min()))-.05
upper=max(float(x.max()),float(y.max()))+.05
axes[0].plot([lower,upper],[lower,upper],color="#9a9e95",lw=1,ls="--")
axes[0].set(xlim=(lower,upper),ylim=(lower,upper),xlabel="Exact signed effect / max |exact effect|",ylabel="First-order estimate / same scale",
    title="A  One checkpoint, all channel pairs")
axes[0].set_aspect("equal",adjustable="box");style(axes[0])
axes[0].text(.04,.95,f"Roman Empire · narrow GCN · seed {e['seed']}\nSpearman {e['spearman_signed']:.4f}",transform=axes[0].transAxes,va="top",fontsize=9)
cell_dots(axes[1],"edges","spearman_signed",(.90,1.04))
axes[1].set(title="B  Agreement across all 30 checkpoints",xlabel="Spearman correlation of signed edge effects")
axes[1].axvline(1,color="#9a9e95",ls="--",lw=1)
fig.subplots_adjust(wspace=.65,bottom=.22,top=.88)
legend(fig);save(fig,"01-attribution-fidelity")

# Figure 2: basis endpoints and the confounded PCA displacement comparison.
for r in records:
    for j,value in enumerate(r["basis"]["rotation_alignments"]):r["basis"][f"rotation_{j}"]=value
fig,axes=plt.subplots(1,2,figsize=(12.6,6.2),gridspec_kw={"width_ratios":[1.2,1]})
dots(axes[0],["Channel"]+[f"Rotation {j+1}" for j in range(5)]+["PCA"],"basis",["channel_alignment"]+[f"rotation_{j}" for j in range(5)]+["pca_alignment"],identity)
axes[0].set(title="A  Alignment survives rotating the basis",xlabel="Alignment improvement over the baseline",xlim=(-.38,.65))
for i,cell in enumerate(cells):
    rows=[r for r in records if r["cell"]==cell]
    ratio=float(np.median([r["basis"]["displacement"]["channel"] for r in rows])/np.median([r["basis"]["displacement"]["pca"] for r in rows]))
    axes[1].hlines(i,1,ratio,color=color[cell],lw=4)
    axes[1].scatter([ratio],[i],c=color[cell],s=38,zorder=3)
    axes[1].annotate(f"{ratio:.2f}×",(ratio,i),xytext=(7,-3),textcoords="offset points")
axes[1].set_xscale("log")
axes[1].set(title="B  PCA moves the layer much less",xlabel="Channel / PCA displacement\nRatio of cell medians · logarithmic axis",xlim=(.9,100),ylim=(3.5,-.5))
axes[1].set_yticks(range(4),labels);axes[1].set_xticks([1,2,5,10,20,50,100],[1,2,5,10,20,50,100]);style(axes[1])
fig.subplots_adjust(wspace=.65,bottom=.18,top=.91);legend(fig);save(fig,"02-basis-and-displacement")

# Figure 3: alignment and output mass are different measurements.
fig,axes=plt.subplots(1,2,figsize=(12.6,4.5))
dots(axes[0],["Change direction", "Shuffle nodes"],"matched",["direction_gain","support_gain"],identity,threshold=.05)
axes[0].set(title="A  Alignment cost of changing one factor",xlabel="Learned minus control alignment",xlim=(-.025,.31))
axes[0].text(.98,.96,"Active directions\nDashed: 0.05 direction threshold",ha="right",va="top",transform=axes[0].transAxes,fontsize=9)
dots(axes[1],["Top 10 per dictionary", "All live features"],"amplification",["median_log2_ratio__top_impact","median_log2_ratio__all_live"],lambda x:2**x,threshold=1.25)
axes[1].set(title="B  Learned directions move output more",xlabel="Output footprint mass: learned / random",xlim=(1,5.5))
axes[1].text(.98,.96,"Identical displacement profile\nDashed: 1.25× threshold",ha="right",va="top",transform=axes[1].transAxes,fontsize=9)
fig.subplots_adjust(wspace=.70,bottom=.25,top=.86);legend(fig);save(fig,"03-direction-support-and-amplification")

# Figure 4: channel-pair concentration and spatial diffuseness are distinct.
fig,axes=plt.subplots(1,2,figsize=(12.6,4.9))
fractions=np.array(data["curve_rank_fractions"])
for cell in cells:
    curves=np.array([r["edges"]["cumulative_mass"] for r in records if r["cell"]==cell])
    axes[0].plot(fractions*100,np.median(curves,axis=0)*100,color=color[cell],lw=2)
axes[0].plot([0,100],[0,100],color="#8f958b",ls="--",lw=1)
axes[0].axvline(5,color="#b3b8af",lw=.8)
axes[0].set(title="A  Influence is concentrated in channel pairs",xlabel="Channel pairs, ranked by edge strength (%)",ylabel="Cumulative edge-strength mass (%)",xlim=(0,100),ylim=(0,102));style(axes[0])
axes[0].text(40,20,"Dashed: uniform reference",color="#747b71",fontsize=9)
cell_dots(axes[1],"edges","participation_ratio_top_edges",(.4,.72))
axes[1].set(title="B  Strong edges act across many nodes",xlabel="Participation ratio over graph nodes\n0 = concentrated · 1 = uniform")
fig.subplots_adjust(wspace=.80,bottom=.25,top=.86);legend(fig);save(fig,"04-concentration-and-diffuseness")

# Figure 5: the right panel is a gain over shuffled attribution, not raw rho.
fig,axes=plt.subplots(1,2,figsize=(12.6,4.6))
dots(axes[0],["Head", "Tail"],"tail",["e7a__active_channels__stratum0","e7a__active_channels__stratum1"],identity,threshold=.05)
axes[0].set(title="A  Across independently trained models",xlabel="Pearson conservation gain over matched null",xlim=(-.06,.32))
axes[0].text(.98,.96,"Dashed: 0.05 threshold",ha="right",va="top",transform=axes[0].transAxes,fontsize=9)
dots(axes[1],["Head", "Tail"],"tail",["e7b__active_channels__stratum0","e7b__active_channels__stratum1"],identity)
axes[1].set(title="B  Exact versus approximate, within a model",xlabel="Spearman agreement gain over shuffled attribution",xlim=(.86,1.075))
axes[1].text(.98,.96,"A difference of correlations\nValues can exceed 1",ha="right",va="top",transform=axes[1].transAxes,fontsize=9)
fig.subplots_adjust(wspace=.38,bottom=.25,top=.86);legend(fig);save(fig,"05-strong-and-weak-edges")

manifest={"source":"figure-data.json","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"outputs":outputs,"note":"Dots are checkpoints, short bars are cell medians, black diamonds are pooled checkpoint medians. No inferential confidence intervals are implied."}
(args.output/"figure-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"figures":5,"files":len(outputs),"output":str(args.output)}))
