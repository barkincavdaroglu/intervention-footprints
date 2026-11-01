#!/usr/bin/env python3
"""Recompute post summaries from saved checkpoint records, without loading models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def summarize(rows):
    values = [row['value'] for row in rows]
    return {
        'count': len(values),
        'median': median(values),
        'positive_count': sum(value > 0 for value in values),
        'negative_count': sum(value < 0 for value in values),
        'groups': {
            cell: {
                'count': len(group := [r['value'] for r in rows if r['cell'] == cell]),
                'median': median(group),
            }
            for cell in sorted({r['cell'] for r in rows})
        },
    }


def alignment_rows(report):
    rows = []
    for row in report['checkpoint_records']:
        effect = row['inference']['class_kl']['active_channels']['candidates']['mask_field']
        if effect['best_control'] != 'magnitude_plus_node_role':
            raise ValueError('Unexpected baseline in saved report')
        value = effect['observed_alignment'] - effect['best_control_alignment']
        if not math.isclose(value, effect['incremental_alignment'], abs_tol=1e-12):
            raise ValueError('Reported increment differs from its constituent correlations')
        rows.append({key: row[key] for key in ('dataset', 'backbone', 'cell', 'seed')} | {'value': value})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optionally save the JSON summary')
    args = parser.parse_args()
    sources = {}

    def load(relative):
        raw = (ROOT / relative).read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    development = alignment_rows(load('artifacts/effect_field_gate.json'))
    expansion = alignment_rows(load('artifacts/effect_field_expansion.json'))
    holdout = alignment_rows(load('artifacts/final_holdout/effect_field.json'))
    joint = load('artifacts/final_holdout/actionability.json')
    repair = load('artifacts/effect_field_repair_interactions_v2.json')
    gnn = [r for r in holdout if r['backbone'] != 'matched_mlp']
    mlp = [r for r in holdout if r['backbone'] == 'matched_mlp']
    matched = {(r['dataset'], r['seed']): r['value'] for r in mlp}
    if len(matched) != len(mlp):
        raise ValueError('Duplicate MLP matching keys')
    contrasts = [r | {'value': r['value'] - matched[(r['dataset'], r['seed'])]} for r in gnn]
    summaries = {
        'development_alignment': summarize([r for r in development if r['backbone'] != 'matched_mlp']),
        'citeseer_alignment': summarize([r for r in expansion if r['dataset'] == 'citationfull_citeseer' and r['backbone'] != 'matched_mlp']),
        'holdout_alignment': summarize(gnn),
        'holdout_mlp_alignment': summarize(mlp),
        'holdout_gnn_minus_mlp': {
            backbone: summarize([r for r in contrasts if r['backbone'] == backbone])
            for backbone in sorted({r['backbone'] for r in contrasts})
        },
        'holdout_joint_ablation': summarize([
            {'cell': r['cell'], 'value': r['contrast']['d_j_kl']}
            for r in joint['checkpoint_records']
        ]),
        'development_repair': summarize([
            {'cell': r['cell'], 'value': r['contrast']['d_int']}
            for r in repair['checkpoint_records']
        ]),
    }
    result = {
        'scope': 'Reaggregation of saved checkpoint-level results, not a model rerun.',
        'source_sha256': sources,
        'summaries': summaries,
    }
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
