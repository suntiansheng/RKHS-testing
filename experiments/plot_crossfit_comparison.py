"""Plot the four-learner comparison using the archived manuscript style."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

LEARNERS = (("gam", "Spline"), ("rf", "Random forest"), ("lasso", "Lasso"), ("dnn", "DNN"))
METHODS = (("full", "Full sample", "#0072B2", "o", "-"),
           ("cf2", "Two-fold cross-fit", "#CC79A7", "D", "--"),
           ("oracle", "Oracle", "#333333", None, (0, (5, 2))))
INTERIM_LABEL = ""


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                        "axes.titlesize":9,"axes.labelsize":9,"xtick.labelsize":8,
                        "ytick.labelsize":8,"legend.fontsize":8,"axes.spines.top":False,
                        "axes.spines.right":False,"pdf.fonttype":42,"axes.linewidth":.6,
                        "lines.linewidth":1.2})


def save(fig, output, name):
    if INTERIM_LABEL:
        fig.suptitle(INTERIM_LABEL,fontsize=9,y=1.035,color=".35")
    fig.savefig(output/f"{name}.pdf",bbox_inches="tight",pad_inches=.04)
    fig.savefig(output/f"{name}.png",bbox_inches="tight",pad_inches=.04,dpi=220)
    plt.close(fig)


def grid(sample_sizes, ylabel):
    count=len(sample_sizes)
    fig,axes=plt.subplots(count,4,figsize=(7.6,2.5 if count==1 else 3.9+1.4*(count-2)),
                          sharex=True,sharey=True,squeeze=False)
    for row,n in enumerate(sample_sizes):
        for col,(_,label) in enumerate(LEARNERS):
            ax=axes[row,col]
            ax.set_title(label+rf", $n={n}$",pad=5)
            ax.set_xlim(-.1,6.1)
            ax.set_xticks((0,2,4,6))
            ax.grid(axis="y",color=".92",lw=.5)
    fig.supylabel(ylabel,x=.012,fontsize=9)
    fig.supxlabel(r"Signal amplitude $a$",y=.165 if count==1 else .115,fontsize=9)
    fig.subplots_adjust(left=.075,right=.995,bottom=.36 if count==1 else .25,
                        top=.85 if count==1 else .92,wspace=.16,hspace=.39)
    return fig,axes


def legend(fig, methods=METHODS):
    handles=[Line2D([0],[0],color=c,marker=m,ls=s,markersize=4,label=label)
             for _,label,c,m,s in methods]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.015),
               ncol=len(handles),frameon=False,handlelength=2.3,columnspacing=1.8)


def power(summary,null,sample_sizes,mode,output):
    fig,axes=grid(sample_sizes,"Empirical rejection rate")
    for row,n in enumerate(sample_sizes):
        for col,(learner,label) in enumerate(LEARNERS):
            ax=axes[row,col]
            for method,_,color,marker,line in METHODS:
                selected=summary[(summary.n==n)&(summary.method==method)]
                if method!="oracle":
                    selected=selected[(selected.learner==learner)&(selected.propensity_mode==mode)]
                selected=selected.sort_values("amplitude")
                ax.plot(selected.amplitude,selected.rejection_rate,color=color,marker=marker,
                        linestyle=line,markersize=3,alpha=.94)
            flagged=null[(null.n==n)&(null.learner==learner)&(null.propensity_mode==mode)
                         &(null.method.isin(["full","cf2"]))&(null.size_screen=="flagged")]
            if len(flagged):
                ax.set_title(label+rf", $n={n}$"+"*",pad=5)
            ax.axhline(.05,color=".55",lw=.6,ls=":")
            ax.set_ylim(-.025,1.035)
            ax.set_yticks((0,.5,1))
    legend(fig)
    save(fig,output,f"simulation_crossfit_additive_{mode}_propensity")


def outcome_error(summary,sample_sizes,output):
    fig,axes=grid(sample_sizes,"Independent outcome MSE")
    data=summary[(summary.propensity_mode=="estimated")&summary.method.isin(["full","cf2"])]
    maximum=float(data.independent_mu_mse.max())*1.08
    for row,n in enumerate(sample_sizes):
        for col,(learner,_) in enumerate(LEARNERS):
            ax=axes[row,col]
            for method,_,color,marker,line in METHODS[:2]:
                selected=data[(data.n==n)&(data.learner==learner)&(data.method==method)].sort_values("amplitude")
                ax.plot(selected.amplitude,selected.independent_mu_mse,color=color,marker=marker,
                        linestyle=line,markersize=3,alpha=.94)
            ax.set_ylim(0,maximum)
    legend(fig,METHODS[:2])
    save(fig,output,"simulation_crossfit_additive_nuisance_error")


def paired_gain(differences,sample_sizes,output):
    fig,axes=grid(sample_sizes,"Power difference")
    data=differences[(differences.propensity_mode=="estimated")&(differences.folds==2)]
    lower=min(-.025,float(data.ci_lower.min())-.015)
    upper=max(.025,float(data.ci_upper.max())+.015)
    for row,n in enumerate(sample_sizes):
        for col,(learner,_) in enumerate(LEARNERS):
            ax=axes[row,col]
            selected=data[(data.n==n)&(data.learner==learner)].sort_values("amplitude")
            ax.errorbar(selected.amplitude,selected.full_minus_crossfit,
                        yerr=[selected.full_minus_crossfit-selected.ci_lower,
                              selected.ci_upper-selected.full_minus_crossfit],
                        color="#0072B2",marker="o",markersize=3,capsize=2,elinewidth=.7)
            ax.axhline(0,color=".55",lw=.6,ls=":")
            ax.set_ylim(lower,upper)
    handles=[Line2D([0],[0],color="#0072B2",marker="o",markersize=4,
                    label="Full sample - two-fold cross-fit; pointwise 95% paired intervals")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.015),frameon=False)
    save(fig,output,"simulation_crossfit_additive_power_difference")


def null_size(null,sample_sizes,output):
    fig,axes=plt.subplots(len(sample_sizes),2,figsize=(6.3,2.8*len(sample_sizes)),sharey=True,squeeze=False)
    labels=[label for _,label in LEARNERS]+["Oracle"]
    ymax=max(.10,float(null.wilson_upper.max())*1.15)
    for row,n in enumerate(sample_sizes):
        for col,mode in enumerate(("estimated","known")):
            ax=axes[row,col]
            for method,_,color,marker,_ in METHODS[:2]:
                selected=null[(null.n==n)&(null.propensity_mode==mode)&(null.method==method)]
                selected=selected.set_index("learner").reindex([learner for learner,_ in LEARNERS])
                x=np.arange(4)+(-.10 if method=="full" else .10)
                ax.errorbar(x,selected.rejection_rate,
                            yerr=[selected.rejection_rate-selected.wilson_lower,
                                  selected.wilson_upper-selected.rejection_rate],
                            color=color,marker=marker,linestyle="none",markersize=4,capsize=3,elinewidth=.8)
                for i,(_,point) in enumerate(selected.iterrows()):
                    if point.size_screen=="flagged":
                        ax.text(x[i],point.wilson_upper+.004,"*",ha="center",va="bottom",color=color)
            oracle=null[(null.n==n)&(null.learner=="oracle")].iloc[0]
            ax.errorbar([4],[oracle.rejection_rate],
                        yerr=[[oracle.rejection_rate-oracle.wilson_lower],[oracle.wilson_upper-oracle.rejection_rate]],
                        color="#333333",marker="s",linestyle="none",markersize=4,capsize=3,elinewidth=.8)
            if oracle.size_screen=="flagged":
                ax.text(4,oracle.wilson_upper+.004,"*",ha="center",va="bottom",color="#333333")
            ax.set_title(("Estimated" if mode=="estimated" else "Known")+rf" propensity, $n={n}$")
            ax.set_xticks(np.arange(5),labels,rotation=15,ha="right")
            ax.set_ylim(0,ymax)
            ax.axhline(.05,color=".55",ls=":",lw=.7)
            ax.grid(axis="y",color=".92",lw=.5)
    fig.supylabel("Null rejection rate",x=.01,fontsize=9)
    legend(fig)
    fig.subplots_adjust(left=.10,right=.995,bottom=.29,top=.87,wspace=.17,hspace=.65)
    save(fig,output,"simulation_crossfit_additive_null_size")


def write_report(summary,null,differences,validation,config,output):
    lines=["**Four-learner full-sample versus two-fold cross-fit simulation**", "",
           "Run status: "+("complete" if validation["complete"] else "INTERIM, not all planned replications complete")+".", "",
           f"Additive design; p=20; n={config['sample_sizes']}; independent balanced outer folds; fixed Gaussian bandwidth squared=40. "
           f"All four outcome learners retain the previous specifications. Each cell uses {config['bootstrap_draws']} centered Gaussian multipliers and nominal level 0.05.", "",
           f"Replication targets: {config['null_replications']} under the null and {config['power_replications']} per positive amplitude. "
           f"Nuisance MSE uses {config['reference_n']:,} independent diagnostic covariates. "
           "All learner curves are retained. A panel asterisk marks a feasible method whose null rate is flagged by the prespecified two-sided Holm screen.", "",
           "Null rejection rates with estimated propensity:", "",
           "| n | Learner | Full sample | Cross-fit | Full screen | Cross-fit screen |",
           "|---:|---|---:|---:|---|---|"]
    for n in config["sample_sizes"]:
        for learner,label in LEARNERS:
            selected=null[(null.n==n)&(null.learner==learner)&(null.propensity_mode=="estimated")].set_index("method")
            if not {"full","cf2"}.issubset(selected.index):
                continue
            full,cf=selected.loc["full"],selected.loc["cf2"]
            lines.append(f"| {n} | {label} | {full.rejection_rate:.3f} | {cf.rejection_rate:.3f} | {full.size_screen} | {cf.size_screen} |")
    lines += ["", "Power at a=2.5 with estimated propensity:", "",
              "| n | Learner | Full sample | Cross-fit | Difference | Paired 95% interval |",
              "|---:|---|---:|---:|---:|---|"]
    selected=differences[(differences.amplitude==2.5)&(differences.propensity_mode=="estimated")&(differences.folds==2)]
    for n in config["sample_sizes"]:
        for learner,label in LEARNERS:
            row=selected[(selected.n==n)&(selected.learner==learner)]
            if row.empty:
                continue
            r=row.iloc[0]
            lines.append(f"| {n} | {label} | {r.full_power:.3f} | {r.crossfit_power:.3f} | {r.full_minus_crossfit:+.3f} | [{r.ci_lower:+.3f}, {r.ci_upper:+.3f}] |")
    lines += ["", "Independent nuisance prediction MSE at a=2.5 (estimated-propensity implementation):", "",
              "| n | Learner | Outcome MSE, full | Outcome MSE, cross-fit | Propensity MSE, full | Propensity MSE, cross-fit |",
              "|---:|---|---:|---:|---:|---:|"]
    for n in config["sample_sizes"]:
        for learner,label in LEARNERS:
            selected=summary[(summary.n==n)&(summary.learner==learner)&(summary.amplitude==2.5)
                             &(summary.propensity_mode=="estimated")].set_index("method")
            if not {"full","cf2"}.issubset(selected.index):
                continue
            full,cf=selected.loc["full"],selected.loc["cf2"]
            lines.append(f"| {n} | {label} | {full.independent_mu_mse:.3f} | {cf.independent_mu_mse:.3f} | "
                         f"{full.independent_e_mse:.5f} | {cf.independent_e_mse:.5f} |")
    lines += ["", "The main power figure uses estimated propensity. The known-propensity figure isolates outcome fitting. "
              "The nuisance figure compares independent outcome MSE; the power-difference figure shows pointwise paired Monte Carlo intervals. "
              "The size figure shows Wilson intervals, with an asterisk for Holm-flagged cells. The oracle is the same statistic in every learner panel.", "",
              "Interpret power differences together with null calibration. Close point estimates or an interval containing zero do not establish equivalence. "
              "These figures provide learner-specific comparisons, without imposing a common ordering.", ""]
    (output/"report.md").write_text("\n".join(lines))


def main():
    global INTERIM_LABEL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-interim",action="store_true")
    args=parser.parse_args()
    validation=json.loads((args.input/"validation.json").read_text())
    if not validation["complete"] and not args.allow_interim:
        raise ValueError("Run is incomplete; use a separate interim output with --allow-interim")
    config=json.loads((args.input/"config.json").read_text())["config"]
    summary=pd.read_csv(args.input/"summary.csv")
    if not validation["complete"]:
        minimum,maximum=int(summary.replications.min()),int(summary.replications.max())
        count=str(minimum) if minimum==maximum else f"{minimum}-{maximum}"
        INTERIM_LABEL=f"Interim simulation: {count} replications per cell"
    null=pd.read_csv(args.input/"null_size.csv")
    differences=pd.read_csv(args.input/"paired_differences.csv")
    sample_sizes=sorted(summary.n.unique())
    args.output.mkdir(parents=True,exist_ok=True)
    style()
    power(summary,null,sample_sizes,"estimated",args.output)
    power(summary,null,sample_sizes,"known",args.output)
    outcome_error(summary,sample_sizes,args.output)
    paired_gain(differences,sample_sizes,args.output)
    null_size(null,sample_sizes,args.output)
    write_report(summary,null,differences,validation,config,args.output)
    (args.output/"figure_metadata.json").write_text(json.dumps({"source":str(args.input.resolve()),
              "complete":validation["complete"],"config":config,"style_reference":"plot_results.py"},indent=2)+"\n")
    print(f"Saved five PDF/PNG figures and report to {args.output}")


if __name__=="__main__":
    main()
