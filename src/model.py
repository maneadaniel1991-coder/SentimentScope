import torch
import torch.nn as nn
from transformers import AutoModel

from .config import Config


class SentimentClassifier(nn.Module):
    """Transformer-based classifier for binary sentiment analysis."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(hidden_size, cfg.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_repr))
        return logits
