#!/usr/bin/env python3
"""Plot collected validation tables using matplotlib; no torch/GPU.

Single-seed results: intentionally no uncertainty/error bars.
"""
import argparse
import csv
from pathlib import Path


def rows(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def value(row, key):
    x = row.get(key)
    return float(x) if x not in (None, "", "None") else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("output exists; choose a new version")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.out.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    tables = args.report / "tables"
    datasets = ["ut-zappos", "mit-states", "cgqa"]
    colors = {"A": "#555555", "C": "#2171B5", "D": "#CB181D", "E": "#7A0177"}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(str(args.out / (name + ".png")), dpi=300, bbox_inches="tight")
        fig.savefig(str(args.out / (name + ".pdf")), bbox_inches="tight")
        plt.close(fig)

    ac = [r for r in rows(tables / "ac_comparison.csv") if r["comparison_status"] == "matched"]
    if ac:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for j, setting in enumerate(["cw", "ow"]):
            for i, (metric, label) in enumerate([("delta_auc_pp", "AUC"), ("delta_hm_pp", "HM")]):
                ax = axes[i, j]
                for k, ds in enumerate(datasets):
                    rr = next((r for r in ac if r["dataset"] == ds and r["setting"] == setting), None)
                    y = value(rr, metric) if rr else None
                    if y is None:
                        ax.text(k, 0, "missing", ha="center", va="bottom", color="gray")
                    else:
                        ax.bar(k, y, color="#2171B5" if y >= 0 else "#CB181D", width=.65)
                        ax.annotate("{:+.2f}".format(y), (k, y), xytext=(0, 4 if y >= 0 else -4), textcoords="offset points", ha="center", va="bottom" if y >= 0 else "top")
                ax.axhline(0, color="black", lw=.7)
                ax.set_xticks(range(3))
                ax.set_xticklabels(datasets)
                ax.set_ylabel("C - A {} (pp)".format(label))
                ax.set_title(setting.upper() + ": paired validation, one seed per dataset")
                ax.margins(y=.3)
        save(fig, "ac_paired_gains")

    vals = [r for r in rows(tables / "all_val_metrics.csv") if r["dataset"] == "ut-zappos" and r["row_status"] == "verified"]
    if vals:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        for j, setting in enumerate(["cw", "ow"]):
            for i, metric in enumerate(["auc_pct", "hm_pct"]):
                ax = axes[i, j]
                for k, variant in enumerate("ACDE"):
                    rr = next((r for r in vals if r["setting"] == setting and r["variant"] == variant), None)
                    y = value(rr, metric) if rr else None
                    if y is not None:
                        ax.bar(k, y, color=colors[variant], width=.65)
                ax.set_xticks(range(4))
                ax.set_xticklabels(list("ACDE"))
                ax.set_ylabel(metric.replace("_pct", "").upper() + " (%)")
                ax.set_ylim(bottom=0)
                ax.set_title("UT-Zappos validation " + setting.upper())
        save(fig, "ut_acde_ablation")

    lambdas = rows(tables / "lambda_analysis.csv")
    if lambdas:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for j, setting in enumerate(["cw", "ow"]):
            for i, metric in enumerate(["auc", "hm"]):
                ax = axes[i, j]
                for label, color in [("Dpost", colors["D"]), ("Epost", colors["E"])]:
                    rr = sorted([r for r in lambdas if r["diagnostic"] == label], key=lambda r: float(r["lambda"]))
                    if rr:
                        ax.plot([float(r["lambda"]) for r in rr], [float(r["delta_" + setting + "_" + metric + "_pp"]) for r in rr], marker="o", label=label, color=color)
                ax.axhline(0, color="gray", lw=.7)
                ax.set_xlabel("lambda (fixed checkpoint)")
                ax.set_ylabel("Delta {} {} (pp)".format(setting.upper(), metric.upper()))
                ax.legend(frameon=False)
        save(fig, "lambda_post_diagnostics")

    curves = [r for r in rows(tables / "epoch_curves.csv") if r["phase"] == "val"]
    if curves:
        fig, axes = plt.subplots(3, 2, figsize=(10, 9))
        for i, ds in enumerate(datasets):
            for variant in "ACDE":
                rr = sorted([r for r in curves if r["dataset"] == ds and r["variant"] == variant], key=lambda r: int(r["epoch"]))
                if rr:
                    axes[i, 0].plot([int(r["epoch"]) for r in rr], [float(r["auc_pct"]) for r in rr], marker=".", color=colors[variant], label=variant)
                    losses = [r for r in rr if value(r, "loss_last_logged_avg") is not None]
                    if losses:
                        axes[i, 1].plot([int(r["epoch"]) for r in losses], [float(r["loss_last_logged_avg"]) for r in losses], marker=".", color=colors[variant], label=variant)
            for ax in axes[i]:
                ax.set_xlabel("epoch")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(frameon=False)
            axes[i, 0].set_ylabel("Validation AUC (%)")
            axes[i, 0].set_title(ds + ": training-config evaluation, rounded log")
            axes[i, 1].set_ylabel("Last logged cumulative training loss")
            axes[i, 1].set_title(ds + ": convergence diagnostic")
        save(fig, "learning_curves")
    print("FIGURES:", args.out)


if __name__ == "__main__":
    main()
