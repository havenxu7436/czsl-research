#!/usr/bin/env python3
"""Plot the independently audited generic-mixture control results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output" / "generic_mixture_controls_results_20261010" / "independent_audit.json"
OUT = ROOT / "output" / "generic_mixture_controls_results_20261010" / "generic_mixture_controls_summary.png"

COLORS = {
    "G_K": "#1f77b4",
    "G_name": "#2ca02c",
    "G_perm1": "#d62728",
    "G_perm2": "#9467bd",
}
LABELS = {
    "G_K": "G_K (correct knowledge)",
    "G_name": "G_name (names only)",
    "G_perm1": "G_perm1 (shuffled)",
    "G_perm2": "G_perm2 (shuffled)",
}


def pct(value: float) -> float:
    return 100.0 * value


def main() -> None:
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = data["records"]
    selected = data["selected"]

    baseline_cw = pct(records["G_K"][0]["metrics_raw"]["cw"]["AUC"])
    baseline_ow = pct(records["G_K"][0]["metrics_raw"]["ow"]["AUC"])
    baseline_attr = pct(records["G_K"][0]["gt_object_attr_accuracy"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ax_cw, ax_ow, ax_delta, ax_attr = axes.flat

    for group in COLORS:
        epochs = [r["epoch"] for r in records[group]]
        cw = [pct(r["metrics_raw"]["cw"]["AUC"]) for r in records[group]]
        ow = [pct(r["metrics_raw"]["ow"]["AUC"]) for r in records[group]]
        ax_cw.plot(epochs, cw, marker="o", ms=3.5, lw=1.7, color=COLORS[group], label=LABELS[group])
        ax_ow.plot(epochs, ow, marker="o", ms=3.5, lw=1.7, color=COLORS[group], label=LABELS[group])

        epoch = selected[group]["epoch"]
        ax_cw.scatter(epoch, pct(selected[group]["metrics_raw"]["cw"]["AUC"]), marker="*", s=125,
                      color=COLORS[group], edgecolor="black", linewidth=0.5, zorder=5)
        ax_ow.scatter(epoch, pct(selected[group]["metrics_raw"]["ow"]["AUC"]), marker="*", s=125,
                      color=COLORS[group], edgecolor="black", linewidth=0.5, zorder=5)

    for ax, baseline, title in (
        (ax_cw, baseline_cw, "Closed-world AUC trajectory"),
        (ax_ow, baseline_ow, "Open-world AUC trajectory"),
    ):
        ax.axhline(baseline, color="#555555", ls="--", lw=1.2, label="Frozen CPF baseline")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC (%)")
        ax.set_xticks(range(0, 11))
        ax.grid(alpha=0.22)

    groups = list(COLORS)
    x = list(range(len(groups)))
    cw_delta = [pct(selected[g]["metrics_raw"]["cw"]["AUC"]) - baseline_cw for g in groups]
    ow_delta = [pct(selected[g]["metrics_raw"]["ow"]["AUC"]) - baseline_ow for g in groups]
    width = 0.36
    ax_delta.bar([v - width / 2 for v in x], cw_delta, width=width, color="#4c78a8", label="CW AUC")
    ax_delta.bar([v + width / 2 for v in x], ow_delta, width=width, color="#f58518", label="OW AUC")
    ax_delta.axhline(0, color="#555555", lw=1)
    ax_delta.set_xticks(x, groups)
    ax_delta.set_ylabel("Selected delta vs baseline (pp)")
    ax_delta.set_title("Selected improvement is not knowledge-specific")
    ax_delta.grid(axis="y", alpha=0.22)
    ax_delta.legend(frameon=False)
    for i, values in enumerate((cw_delta, ow_delta)):
        offset = -width / 2 if i == 0 else width / 2
        for xi, value in zip(x, values):
            ax_delta.text(xi + offset, value + 0.04, f"{value:+.2f}", ha="center", va="bottom", fontsize=8)

    selected_attr = [pct(selected[g]["gt_object_attr_accuracy"]) for g in groups]
    bars = ax_attr.bar(x, selected_attr, color=[COLORS[g] for g in groups])
    ax_attr.axhline(baseline_attr, color="#555555", ls="--", lw=1.2, label=f"Baseline {baseline_attr:.2f}%")
    ax_attr.set_xticks(x, groups)
    ax_attr.set_ylabel("GT-object attribute accuracy (%)")
    ax_attr.set_title("Attribute accuracy does not improve")
    ax_attr.set_ylim(min(selected_attr) - 0.25, baseline_attr + 0.18)
    ax_attr.grid(axis="y", alpha=0.22)
    ax_attr.legend(frameon=False)
    for bar, value in zip(bars, selected_attr):
        ax_attr.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
                     ha="center", va="bottom", fontsize=8)

    handles, labels = ax_cw.get_legend_handles_labels()
    fig.suptitle("Generic conditional adapter with fixed 25% probability mixture", fontsize=14, y=0.985)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.947), ncol=5, frameon=False,
               fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.0, w_pad=1.7)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
