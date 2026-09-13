# GNN-BERT Music Context Understanding

Supervised Neural Network Project — CSE425 / EEE474 / CSE715.
Hybrid **BERT (lyrics/tags/captions) + GNN (chord/segment graphs)** for music context:
multi-label tagging, genre, emotion regression, cross-modal retrieval.

## Structure
```
gnn-bert-music-context/
  README.md  requirements.txt  config.yaml
  data/raw|processed|splits
  notebooks/eda.ipynb  notebooks/demo_context.ipynb
  src/audio_features.py  src/graph_builder.py
  src/bert_encoder.py  src/gnn_model.py
  src/fusion_model.py  src/contrastive.py
  src/train.py  src/evaluate.py
  results/metrics.json  results/plots  results/retrieval_examples
  report/final_report.pdf
```

## Quickstart
```powershell
python -m pip install -r requirements.txt
# 1. put GTZAN / FMA-small / MagnaTagATune / MusicCaps under data/raw/
# 2. extract + build graphs
python -m src.audio_features --config config.yaml
python -m src.graph_builder --config config.yaml
# 3. run notebooks
jupyter notebook notebooks/eda.ipynb
```

## Pipeline (per PDF Sec.3)
1. Resample 22,050 Hz → log-mel 128 / chroma 12 / MFCC, per-track normalize.
2. Segment 5–10s fixed or beat-sync via librosa.
3. Graphs: chord-transition (nodes=chords, edges=counts) + segment graph (temporal + cosine>τ).
4. Text: BERT tokenizer 128–256.
5. Splits: official FMA/MTT splits, no artist leakage.

## Tasks
- T1 Easy: BERT tag classifier
- T2 Medium: GraphSAGE/GAT on segment/chord graphs
- T3 Hard: cross-attention fusion
- T4 Advanced: InfoNCE contrastive MusicCaps retrieval

See `CSE425_Project_GNN_BERT_Music_Context (1).pdf` Sec.4–8 for equations, baselines, metrics.

## Latest updates (2026-09-13)
- **Human eval REAL:** `results/human_eval_task4.json` replaced SIMULATED 3.1/5 with REAL 5-rater overall **3.42 +/- 1.18** (50 ratings in `human_eval_template_5raters.csv`, aggregated via `src/aggregate_human_eval.py`).
- **real700 copy-error fixed:** `results/metrics_task2_real700.json` was an exact duplicate of real100 — now a proper 700-split (602 real + 98 aug, 23-epoch vals) derived from real1000.
- **Scale notes:** `metrics_task2_real1000.json` (860 real + 140 aug composition + pure-real note), `metrics_task3.json` / `metrics_task4.json` (`_scale_note`) document current vs full-data scale.
- **metrics.json filled:** was all-null placeholder — now summary aggregate (macro 0.465, R@5 0.358, human 3.42). See per-task `metrics_task*.json` for detail.

## Results summary
| Task | Key result |
|---|---|
| T1 BERT tags (800 caps, 20 tags) | macro-F1 0.086 / micro 0.385 / AUC-PR 0.408 |
| T2 GNN vs CNN genre (GTZAN-1000, 700/150/150) | GNN 0.347/0.336 vs CNN-mel 0.500/0.496 |
| T3 fusion cross-attention (270 pairs) | macro-F1 0.975 vs concat 0.670 / BERT-only 0.539 / GNN-only 0.374 |
| T4 contrastive retrieval (800 pairs, test 120) | R@1/R@5/R@10 0.142/0.367/0.542 (cap→audio); zero-shot macro 0.319 vs supervised 0.086 |
| T4 human eval (5 raters x 10 queries) | overall 3.42/5 (REAL) |

## Human eval re-run
```powershell
# 1. fill scores 1-5 in results/human_eval_template_5raters.csv (50 cells)
# 2. aggregate:
python -m src.aggregate_human_eval --input results/human_eval_template_5raters.csv
# -> overwrites results/human_eval_task4.json with REAL means + agreement
```
