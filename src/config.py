from dataclasses import dataclass
import torch


@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    max_len: int = 256          # full-length reviews
    batch_size: int = 16        # safe default for GPU
    num_epochs: int = 3         # adjust in Udacity workspace if needed
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    num_labels: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
