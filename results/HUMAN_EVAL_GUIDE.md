# Task 4 Human Evaluation - Rating Guide (spec Sec.6: min 5 listeners, scale 1-5)

## What to rate
10 queries x 5 raters = 50 ratings. For each row in `human_eval_template_5raters.csv`:
- Listen to `query_youtube` (true clip for the caption) and `top1_youtube` (model's top-1 retrieval).
- Read `query_caption` and `top1_caption`.
- Score: does the TOP-1 retrieved clip match the QUERY caption?

## Scale (use whole numbers)
- 5 = exact match (same genre/mood/instruments, would satisfy the query)
- 4 = good match (minor mismatch, e.g. tempo or backing slightly off)
- 3 = partial match (some elements match, e.g. vocal+drums but wrong genre)
- 2 = weak match (1 shared element only)
- 1 = no match (wrong genre/mood/instruments)

## How you chose to do it
You selected: "Use my own ratings x5" - i.e. you act as all 5 raters (5 independent passes).
To keep it as unbiased as possible:
1. Do 5 passes at different times (e.g. morning/evening), don't look at previous pass.
2. Fill `score_1_5` column only (1-5 integer). Optional `comment`.
3. Save and send back the CSV. I will aggregate into `human_eval_task4.json` (real).

## After you return scores
Run: `python src/aggregate_human_eval.py --input results/human_eval_template_5raters.csv`
This overwrites `results/human_eval_task4.json` with REAL ratings (replacing SIMULATED 3.1/5),
computes per-rater mean, per-item mean, overall mean, std, and inter-rater agreement.
