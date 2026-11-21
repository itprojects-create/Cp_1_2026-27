"""
Fine-tuning script for IDRiD disease grading.

Trains a shared image encoder with two classification heads:
- DR severity (5 classes)
- DME severity (4 classes)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from idrid_dataloader import IDRiDDataModule


def set_seed(seed: int) -> None:
    """Make training more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    dataset_root: str = "./preprocessed_idrid"
    output_dir: str = "./outputs/idrid_finetune"
    model_name: str = "resnet50"
    pretrained: bool = True
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    augment_strength: str = "medium"
    val_split: float = 0.15
    seed: int = 42
    freeze_backbone_epochs: int = 1
    dr_loss_weight: float = 1.0
    dme_loss_weight: float = 1.0
    early_stopping_patience: int = 5


class MultiTaskIDRiDModel(nn.Module):
    """Backbone + separate heads for DR and DME classification."""

    def __init__(self, model_name: str = "resnet50", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.model_name = model_name.lower()

        if self.model_name == "resnet18":
            backbone = self._load_resnet18(pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif self.model_name == "resnet50":
            backbone = self._load_resnet50(pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif self.model_name == "efficientnet_b0":
            backbone = self._load_efficientnet_b0(pretrained)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. Choose from: resnet18, resnet50, efficientnet_b0."
            )

        self.dropout = nn.Dropout(dropout)
        self.dr_head = nn.Linear(in_features, 5)
        self.dme_head = nn.Linear(in_features, 4)

    @staticmethod
    def _build_backbone(factory, default_weights, pretrained: bool):
        if not pretrained:
            return factory(weights=None)

        try:
            return factory(weights=default_weights)
        except Exception as error:
            print(
                f"Pretrained weights could not be loaded ({error}). "
                "Continuing with randomly initialized weights."
            )
            return factory(weights=None)

    @classmethod
    def _load_resnet18(cls, pretrained: bool):
        return cls._build_backbone(models.resnet18, models.ResNet18_Weights.DEFAULT, pretrained)

    @classmethod
    def _load_resnet50(cls, pretrained: bool):
        return cls._build_backbone(models.resnet50, models.ResNet50_Weights.DEFAULT, pretrained)

    @classmethod
    def _load_efficientnet_b0(cls, pretrained: bool):
        return cls._build_backbone(
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT,
            pretrained,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = self.dropout(features)
        return {
            "dr_logits": self.dr_head(features),
            "dme_logits": self.dme_head(features),
        }


def freeze_backbone(model: MultiTaskIDRiDModel, freeze: bool) -> None:
    """Freeze or unfreeze shared feature extractor parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def build_dataloaders(config: TrainConfig) -> Tuple[torch.utils.data.DataLoader, ...]:
    data_module = IDRiDDataModule(
        dataset_root=config.dataset_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=True,
        augment_strength=config.augment_strength,
        task="disease_grading",
        val_split=config.val_split,
    )
    return (
        data_module.get_train_loader(),
        data_module.get_val_loader(),
        data_module.get_test_loader(),
    )


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion_dr: nn.Module,
    criterion_dme: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "dr_loss": 0.0,
        "dme_loss": 0.0,
        "dr_acc": 0.0,
        "dme_acc": 0.0,
        "joint_acc": 0.0,
    }
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        dr_labels = batch["dr_grade"].to(device, non_blocking=True)
        dme_labels = batch["dme_grade"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            dr_loss = criterion_dr(outputs["dr_logits"], dr_labels)
            dme_loss = criterion_dme(outputs["dme_logits"], dme_labels)
            loss = config.dr_loss_weight * dr_loss + config.dme_loss_weight * dme_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        dr_predictions = torch.argmax(outputs["dr_logits"], dim=1)
        dme_predictions = torch.argmax(outputs["dme_logits"], dim=1)
        joint_accuracy = (
            ((dr_predictions == dr_labels) & (dme_predictions == dme_labels))
            .float()
            .mean()
            .item()
        )

        totals["loss"] += loss.item()
        totals["dr_loss"] += dr_loss.item()
        totals["dme_loss"] += dme_loss.item()
        totals["dr_acc"] += accuracy_from_logits(outputs["dr_logits"], dr_labels)
        totals["dme_acc"] += accuracy_from_logits(outputs["dme_logits"], dme_labels)
        totals["joint_acc"] += joint_accuracy
        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def evaluate_test_set(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    dr_correct = 0
    dme_correct = 0
    joint_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            dr_labels = batch["dr_grade"].to(device, non_blocking=True)
            dme_labels = batch["dme_grade"].to(device, non_blocking=True)

            outputs = model(images)
            dr_predictions = torch.argmax(outputs["dr_logits"], dim=1)
            dme_predictions = torch.argmax(outputs["dme_logits"], dim=1)

            dr_correct += (dr_predictions == dr_labels).sum().item()
            dme_correct += (dme_predictions == dme_labels).sum().item()
            joint_correct += (
                ((dr_predictions == dr_labels) & (dme_predictions == dme_labels))
                .sum()
                .item()
            )
            total += dr_labels.size(0)

    denominator = max(total, 1)
    return {
        "dr_test_accuracy": dr_correct / denominator,
        "dme_test_accuracy": dme_correct / denominator,
        "joint_test_accuracy": joint_correct / denominator,
    }


def save_history(history: list[Dict[str, float]], output_dir: Path) -> None:
    history_path = output_dir / "training_history.csv"
    if not history:
        return

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(config)
    sample_dataset = train_loader.dataset
    dr_class_weights = sample_dataset.get_class_weights("dr").to(device)
    dme_class_weights = sample_dataset.get_class_weights("dme").to(device)

    model = MultiTaskIDRiDModel(
        model_name=config.model_name,
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)

    criterion_dr = nn.CrossEntropyLoss(weight=dr_class_weights)
    criterion_dme = nn.CrossEntropyLoss(weight=dme_class_weights)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        freeze_backbone(model, freeze=epoch <= config.freeze_backbone_epochs)

        train_metrics = run_epoch(
            model, train_loader, device, criterion_dr, criterion_dme, optimizer, config
        )
        val_metrics = run_epoch(
            model, val_loader, device, criterion_dr, criterion_dme, None, config
        )
        scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dr_acc={val_metrics['dr_acc']:.4f} | "
            f"val_dme_acc={val_metrics['dme_acc']:.4f} | "
            f"val_joint_acc={val_metrics['joint_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    save_history(history, output_dir)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_test_set(model, test_loader, device)
    summary = {
        "config": asdict(config),
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
    }

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("Training finished.")
    print(json.dumps(summary, indent=2))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune an IDRiD disease grading model.")
    parser.add_argument("--dataset-root", default="./preprocessed_idrid")
    parser.add_argument("--output-dir", default="./outputs/idrid_finetune")
    parser.add_argument(
        "--model-name",
        default="resnet50",
        choices=["resnet18", "resnet50", "efficientnet_b0"],
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet-pretrained weights if weights are not cached locally.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--augment-strength",
        default="medium",
        choices=["light", "medium", "heavy"],
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--dr-loss-weight", type=float, default=1.0)
    parser.add_argument("--dme-loss-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    args = parser.parse_args()

    return TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        augment_strength=args.augment_strength,
        val_split=args.val_split,
        seed=args.seed,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        dr_loss_weight=args.dr_loss_weight,
        dme_loss_weight=args.dme_loss_weight,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    train(parse_args())
