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
