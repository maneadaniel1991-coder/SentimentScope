import torch
from sklearn.metrics import classification_report, confusion_matrix

from .config import Config
from .data import create_dataloaders
from .model import SentimentClassifier


@torch.no_grad()
def evaluate_on_test(cfg: Config, max_batches: int | None = None):
    """
    Evaluate the best saved model on the IMDB test set.

    If max_batches is provided, only the first N batches are evaluated.
    For the final project report, call this WITHOUT max_batches.
    """
    _, _, test_loader = create_dataloaders(cfg)

    model = SentimentClassifier(cfg).to(cfg.device)
    model.load_state_dict(torch.load("best_model.pt", map_location=cfg.device))
    model.eval()

    all_labels = []
    all_preds = []

    for i, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        if max_batches is not None and (i + 1) >= max_batches:
            break

    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return {"report": report, "confusion_matrix": cm}
