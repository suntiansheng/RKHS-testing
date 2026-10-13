#!/usr/bin/env python3
"""Plot saved baseline comparisons in the existing CaSiNo figure style."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot as existing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=Path(__file__).resolve().parents[1]/"runs/casino-baseline")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    effects=pd.read_csv(args.results_dir/"standardized_contrasts.csv")
    tests=pd.read_csv(args.results_dir/"baseline_tests.csv")
    tests=tests.loc[(tests["sample"]=="frozen_main") & (tests.outcome=="mean_satisfaction") & (tests.bandwidth==1)]
    existing.configure_style()
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.65),gridspec_kw={"width_ratios":[1,1,1.15]})
    colors={"unadjusted":"#444444",**existing.LEARNER_COLORS}
    methods=["unadjusted","ridge","lasso","random_forest"]
    labels=["Unadjusted","Ridge","Lasso","Forest"]
    for ax,outcome,title,scale,xlabel in [
        (axes[0],"mean_satisfaction","(a) Mean satisfaction",1,"Difference in rating points"),
        (axes[1],"both_extremely_satisfied","(b) Both maximally satisfied",100,"Difference in percentage points"),
    ]:
        subset=effects.loc[effects.outcome==outcome].set_index("method")
        for pos,method in enumerate(methods):
            row=subset.loc[method]
            if not np.isfinite([row.estimate,row.ci_lower,row.ci_upper]).all():
                raise ValueError("Nonfinite contrast")
            point=scale*row.estimate
            ax.errorbar(point,pos,xerr=[[scale*(row.estimate-row.ci_lower)],[scale*(row.ci_upper-row.estimate)]],
                fmt="o",color=colors[method],markersize=4,capsize=2,elinewidth=1)
        ax.axvline(0,color="#888888",lw=0.8,ls="--")
        ax.set_yticks(range(4),labels)
        ax.set_ylim(3.5,-0.5)
        ax.set_title(title,loc="left",fontsize=8.5)
        ax.set_xlabel(xlabel,fontsize=7.5)
        ax.grid(axis="x",alpha=0.15)
    ax=axes[2]
    for pos,learner in enumerate(methods[1:]):
        for regime,offset,marker,label in [("no_split",-0.13,"o","Full sample"),("crossfit_5fold",0.13,"s","Cross-fit, K=5")]:
            row=tests.loc[(tests.learner==learner)&(tests.regime==regime)].iloc[0]
            ax.scatter(row.p_value,pos+offset,marker=marker,s=24,color=colors[learner],
                facecolors=colors[learner] if regime=="no_split" else "none",linewidths=1,
                label=label if pos==0 else None)
    ax.set_xscale("log")
    ax.axvline(0.05,color="#777777",lw=0.8,ls="--")
    ax.set_xlim(1/5001/1.5,1)
    ax.set_xticks([1/5001,0.001,0.01,0.05,1],["1/5001",".001",".01",".05","1"])
    ax.tick_params(axis="x",labelsize=6.5)
    ax.set_yticks(range(3),labels[1:])
    ax.set_ylim(2.65,-0.55)
    ax.set_title("(c) Baseline regression test",loc="left",fontsize=8.5)
    ax.set_xlabel("Multiplier p-value",fontsize=7.5)
    ax.legend(loc="lower right",fontsize=6.8,handletextpad=0.3)
    fig.subplots_adjust(left=0.075,right=0.995,bottom=0.23,top=0.86,wspace=0.55)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,bbox_inches="tight",pad_inches=0.04)
    fig.savefig(args.output.with_suffix(".png"),dpi=220,bbox_inches="tight",pad_inches=0.04)
    plt.close(fig)
    print(args.output)


if __name__=="__main__":
    main()
