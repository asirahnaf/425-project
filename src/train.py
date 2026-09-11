"""Unified training entrypoint (wires Tasks 1-4). Implemented in next steps."""
from __future__ import annotations
import argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["1","2","3","4"], default="2")
    ap.add_argument("--config", default="config.yaml")
    print("train stub — Task", ap.parse_args().task)
