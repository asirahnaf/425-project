"""Aggregate REAL 5-rater human eval into human_eval_task4.json.
Usage:
  python -m src.aggregate_human_eval --input results/human_eval_template_5raters.csv
Input CSV columns: rater,query_id,query_caption,top1_track,top1_caption,query_youtube,top1_youtube,score_1_5,comment
Output: results/human_eval_task4.json (REAL, replaces SIMULATED) with means + agreement.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/human_eval_template_5raters.csv')
    ap.add_argument('--output', default='results/human_eval_task4.json')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    inp = root/args.input if not Path(args.input).is_absolute() else Path(args.input)
    outp = root/args.output if not Path(args.output).is_absolute() else Path(args.output)

    scores_by_rater = {}
    comments = []
    with open(inp, encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get('score_1_5','').strip()
            if s == '':
                continue
            try:
                v = int(float(s))
            except:
                raise ValueError(f"bad score {s} row {row}")
            if not 1 <= v <= 5:
                raise ValueError(f'score out of range 1-5: {v}')
            rat = row['rater'].strip()
            qid = int(row['query_id'])
            scores_by_rater.setdefault(rat, {})[qid] = v
            if row.get('comment','').strip():
                comments.append({'rater': rat, 'query_id': qid, 'comment': row['comment'].strip()})

    if len(scores_by_rater) < 5:
        print(f'WARNING: only {len(scores_by_rater)} raters found, need 5. Found: {sorted(scores_by_rater)}')
    # check each rater has 10
    for rat, d in scores_by_rater.items():
        if len(d) != 10:
            print(f'WARNING: {rat} has {len(d)}/10 ratings, missing {[i for i in range(10) if i not in d]}')

    raters_sorted = sorted(scores_by_rater.keys())
    mat = np.array([[scores_by_rater[rat].get(i, np.nan) for i in range(10)] for rat in raters_sorted], dtype=float)
    # require complete
    if np.isnan(mat).any():
        missing = np.argwhere(np.isnan(mat))
        print(f'ERROR: missing scores at (rater_idx, query): {missing.tolist()} - fill all 50 cells')
        return

    per_rater_mean = mat.mean(axis=1).tolist()
    per_item_mean = mat.mean(axis=0).tolist()
    per_item_std = mat.std(axis=0, ddof=1).tolist() if len(raters_sorted) > 1 else [0.0]*10
    overall_mean = float(mat.mean())
    overall_std = float(mat.std(ddof=1))

    # simple agreement: % of items where max-min <=1
    agree = float(np.mean([mat[:, i].max() - mat[:, i].min() <= 1 for i in range(10)]))

    raters_out = [{'rater': rat, 'scores': [int(x) for x in mat[j].tolist()], 'mean': round(float(per_rater_mean[j]), 2)} for j, rat in enumerate(raters_sorted)]

    out = {
        'note': 'REAL human ratings (5 raters x 10 queries). Scale 1=no match to 5=exact match. Collected by student (5 passes).',
        'n_raters': len(raters_sorted),
        'n_items': 10,
        'raters': raters_out,
        'per_item_mean': [round(float(x), 2) for x in per_item_mean],
        'per_item_std': [round(float(x), 2) for x in per_item_std],
        'overall_mean': round(overall_mean, 2),
        'overall_std': round(overall_std, 2),
        'agreement_within1': round(agree, 3),
        'comments': comments,
    }
    outp.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'wrote {outp}')
    print(f'overall {out["overall_mean"]} +/- {out["overall_std"]}, agreement(<=1): {out["agreement_within1"]}')
    print(f'per-item mean: {out["per_item_mean"]}')

if __name__ == '__main__':
    main()
