from __future__ import annotations

import os

import torch

from src.config.load import load_evaluation_run
from src.eval.cross_dataset import evaluate_cross_dataset


def configure_cpu_threads() -> None:
    requested = os.environ.get("OMP_NUM_THREADS", "").strip()
    thread_count = int(requested) if requested.isdigit() and int(requested) > 0 else 4
    torch.set_num_threads(thread_count)
    print(f"[cross-eval] torch CPU threads={torch.get_num_threads()}", flush=True)


def main() -> None:
    configure_cpu_threads()
    config = load_evaluation_run()
    evaluate_cross_dataset(config)


if __name__ == "__main__":
    main()
