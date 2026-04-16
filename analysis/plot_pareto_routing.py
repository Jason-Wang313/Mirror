"""Generate fig3_pareto_routing.pdf — the headline Pareto plot for Finding 3.

Reads the instance-baseline summary JSON and plots weak-domain CFR vs.
escalation budget for MIRROR domain routing and instance-level baselines.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'data' / 'results' / 'exp9_instance_baselines' / \
    'v20_maximal_closure_20260326T2200Z_instance_c1_all_paradigms' / \
    'instance_baseline_summary.json'
OUT = REPO / 'paper' / 'figures' / 'fig3_pareto_routing.pdf'


def frontier_xy(points):
    """Convert autonomy-target frontier points to (escalation, weak_cfr)."""
    pts = sorted(points, key=lambda p: p['autonomy_target'])
    return ([1.0 - p['mean_autonomy_rate'] for p in pts],
            [p['mean_weak_cfr'] for p in pts])


def main():
    with SRC.open() as f:
        d = json.load(f)

    macro = d['macro_summary']
    frontier = d['frontier_macro_summary']

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # Baseline frontiers
    style = {
        'confidence_threshold_frontier': dict(
            color='#2E86AB', marker='o', linestyle='-',
            label='Confidence threshold'),
        'self_consistency_frontier': dict(
            color='#F77F00', marker='s', linestyle='-',
            label='Self-consistency'),
        'calibrated_confidence_frontier': dict(
            color='#6A994E', marker='^', linestyle='-',
            label='Calibrated confidence (Platt)'),
    }
    for key, kw in style.items():
        xs, ys = frontier_xy(frontier[key])
        ax.plot(xs, ys, linewidth=1.8, markersize=5.5, **kw)

    # MIRROR domain routing — the headline point
    mr = macro['mirror_domain_routing']
    ax.scatter([mr['mean_escalation_rate']], [mr['mean_weak_cfr']],
               s=220, marker='*', color='#D62828', zorder=5,
               edgecolors='black', linewidths=1.1,
               label=r'$\mathbf{MIRROR\ domain\ routing\ (C4)}$')

    # Conformal (high-escalation comparison)
    conf = macro['conformal_style_budget_matched']
    ax.scatter([conf['mean_escalation_rate']], [conf['mean_weak_cfr']],
               s=90, marker='D', color='#6D597A', zorder=4,
               edgecolors='black', linewidths=0.8,
               label='Conformal-style thresholding')

    # No-routing baseline
    nr = macro['no_routing']
    ax.scatter([nr['mean_escalation_rate']], [nr['mean_weak_cfr']],
               s=90, marker='v', color='#8D99AE', zorder=3,
               edgecolors='black', linewidths=0.6,
               label='No routing (C1 baseline)')

    # Matched-budget vertical reference line
    ax.axvline(mr['mean_escalation_rate'], color='gray',
               linestyle=':', linewidth=1, alpha=0.7)
    ax.text(mr['mean_escalation_rate'] + 0.01, 0.62,
            f"matched budget\n({mr['mean_escalation_rate']*100:.1f}% escalation)",
            fontsize=8.5, color='gray', va='top')

    # Annotations for budget-matched instance baselines on the ref line
    bm = [
        ('confidence_threshold_budget_matched', '31.7%'),
        ('self_consistency_budget_matched', '32.3%'),
        ('calibrated_confidence_budget_matched', '30.2%'),
    ]
    for key, _label in bm:
        pt = macro[key]
        ax.scatter([pt['mean_escalation_rate']], [pt['mean_weak_cfr']],
                   s=50, marker='x', color='black', zorder=5)

    ax.set_xlabel('Escalation budget (fraction of weak-domain components routed out)',
                  fontsize=10)
    ax.set_ylabel('Weak-domain CFR (mean over 16 models)', fontsize=10)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 0.75)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={'Author': '', 'Creator': '', 'Producer': '',
                               'Title': '', 'Subject': '', 'Keywords': ''})
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
