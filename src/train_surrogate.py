from __future__ import annotations

import os

import torch

from src.config.load import load_training_run
from src.train.trainer import train_surrogate


def configure_cpu_threads() -> None:
    requested = os.environ.get("OMP_NUM_THREADS", "").strip()
    thread_count = int(requested) if requested.isdigit() and int(requested) > 0 else 4
    torch.set_num_threads(thread_count)
    print(f"[training] torch CPU threads={torch.get_num_threads()}", flush=True)


def main() -> None:
    configure_cpu_threads()
    config = load_training_run()
    train_surrogate(config)


if __name__ == "__main__":
    main()