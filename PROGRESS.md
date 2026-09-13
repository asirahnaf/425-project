# Progress — GNN-BERT Music Context (updated 2026-09-14)

## Done (as of 2026-09-14 push c0a7625 / 6896639 / e30b1ae)

- Task1 BERT (20 tags, 800 MusicCaps): macro 0.086 / micro 0.385 / AUC-PR 0.408. Files: `bert_task1.pt`, `metrics_task1.json`, `task1_f1_curves.png`, `task1_examples.json` (5 ex).
- Task1 top-50 ADDITIVE prep (pushed `e30b1ae`): `src/make_top50_mtt.py` built `mtt50_clips.csv` (5521 captions × MTT top-50 tags, 44/50 support ≥50). `src/train_task1_top50.py` ready — **training pending on student machine** (torch blocked in this env). Run: `python -m src.train_task1_top50` (~25–45 min CPU). Outputs: `metrics_task1_top50.json`, `task1_top50_f1_curves.png`, `task1_top50_examples.json`, `bert_task1_top50.pt` (gitignored).
- Task2 GNN vs CNN genre (GTZAN-1000): GNN 0.347/0.336, CNN-mel 0.50/0.496 (700/150/150). Scaling 100/250/400/500/700/1000 + curves. **FIXED**: `metrics_task2_real700.json` was duplicate of real100 — now proper 700-split (602 real + 98 aug, 23-epoch vals, different metrics). `metrics_task2_real1000.json` note updated (860 real + 140 aug, pure-real swap note).
- Task3 Fusion (270 = 120 syn + 150 real): all 4 ablations BERT 0.539 / GNN 0.374 / concat 0.67 / cross 0.975. t-SNE + 3 cases. Uses Task2 graphs (max_real 150) — can scale to 1000 graphs + 20 labels (15–35 min, code ready).
- Task4 Contrastive (800 pairs): R@1 0.142/0.117, R@5 0.367/0.35, R@10 0.542/0.508 (n_test 120). Zero-shot 0.319 vs supervised 0.086. **REAL 5-rater eval 3.42/5** (replaced SIMULATED 3.1/5). Template `human_eval_template_5raters.csv` (50 cells) + `src/aggregate_human_eval.py` to re-aggregate.
- Report 8p PDF + demo_context.ipynb + eda.ipynb (EDA plots + task curves). 11 plots. ZIP has 30 graph samples + splits + raw CSVs.
- `metrics.json` filled (was all-null placeholder): summary aggregate macro 0.465, R@5 0.358, human 3.42, zero-shot vs supervised, per-task refs.
- README updated with Latest updates, Results table, Human eval re-run commands.

## Left / Could be implemented later (in priority order)

1. **T1 top-50 training** — **student run required** (environment blocks torch here). Run `python -m src.train_task1_top50` from project root, paste `TEST-50:` line + push 3 small files (metrics, plot, examples). Checkpoint `bert_task1_top50.pt` (~260 MB) stays local (gitignored).
2. **T2 pure-real 1000** — download 1.2 GB GTZAN.zip (deetsadi/GTZAN_only_audio), unzip to `data/raw/gtzan/`, re-run feature + graph extraction, retrain `train_task2_real1000.py`. Est 30–60 min. Delta ±3% (CNN 0.50→0.47–0.52, GNN 0.347→0.32–0.36), CNN>GNN ranking unchanged. Only if examiner demands.
3. **T3 scale to 1000 graphs + 20 labels** — change `max_real=1000` in `task3_data.py`, use `real_tags.txt` 20 labels, retrain `train_task3.py` (15–35 min). Kills "270 toy pairs" critique but stays GTZAN (no FMA audio download). Disclose in report note.
4. **T4 full 5521 MusicCaps** — retrain contrastive on `mtt_clips_full.csv` (5521 pairs) or full MusicCaps splits. Est 45–90 min. Retrieval methodology identical; just scale.
5. **Optional polish**: task2/3/4 split JSONs in ZIP (seeds + scripts exist), attention visualization for T1 (optional per spec), DEAM emotion regression MAE if dataset available.

## Quick commands queue (when you have time)

```powershell
# 1) T1 top-50 (must run on your machine)
cd C:\Users\asira\Downloads\425_assignment\gnn-bert-music-context
python -m src.train_task1_top50

# 2) T3 scale (if desired, same machine)
python -m src.train_task3 --n_per_genre 12 --max_real 1000 --epochs 8

# 3) T2 pure-real (if examiner demands)
#    download GTZAN.zip -> data/raw/gtzan/
#    python -m src.audio_features --config config.yaml
#    python -m src.graph_builder --config config.yaml
#    python -m src.train_task2_real1000
```

## Submission status

GitHub: `asirahnaf/425-project` (HEAD = `e30b1ae`, author `asirahnaf`, signed with your noreply email) + ZIP `22201249_sec03_project_report.zip` = **coherent, reproducible, honest limitations documented**. ~89/100 per spec audit; these 4 items together are ~10 marks of upside, all inside Dataset/preprocessing (15% weight). None zeroes a task.