from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import Config
from .data import create_dataloaders
from .model import SentimentClassifier


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def eval_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "Validation",
) -> Tuple[float, float]:
    """Evaluate for one epoch and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=phase, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def train_model(cfg: Config) -> None:
    """Train the sentiment classifier, saving the best model by validation accuracy."""
    train_loader, val_loader, _ = create_dataloaders(cfg)
    device = torch.device(cfg.device)

    model = SentimentClassifier(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    best_model_path = "best_model.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, phase="Validation")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch}/{cfg.num_epochs} "
            f"- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    if best_acc >= 0.0:
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
