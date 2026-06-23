from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ccr_multiclass import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    FraudMulticlassDataset,
    MulticlassCCRClassifier,
    MulticlassCCRDataCollator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练八分类 CCR：Stage 1 特征解耦 + Stage 2 IPW 与因果约束"
    )

    parser.add_argument(
        "--train-file",
        default="data/processed_multiclass/train.csv",
    )
    parser.add_argument(
        "--val-file",
        default="data/processed_multiclass/val.csv",
    )
    parser.add_argument(
        "--test-file",
        default="data/processed_multiclass/test.csv",
    )
    parser.add_argument(
        "--model-name",
        default="hfl/chinese-roberta-wwm-ext",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ccr_multiclass_run",
    )

    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--feature-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage1-lr", type=float, default=2e-5)
    parser.add_argument("--stage2-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument(
        "--cov-lambda",
        type=float,
        default=0.01,
        help="Stage 1 协方差正则系数",
    )
    parser.add_argument(
        "--causal-lambda",
        type=float,
        default=1.0,
        help="Stage 2 反事实因果约束系数",
    )
    parser.add_argument(
        "--counterfactual-chunk-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-ipw",
        type=float,
        default=10.0,
        help="限制极少数组的最大 IPW，避免训练不稳定",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["balanced", "none"],
        default="balanced",
        help="Stage 1 是否使用类别平衡权重",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help='例如 "cuda"、"cuda:0" 或 "cpu"；默认自动选择',
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 环境下启用混合精度",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def move_model_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    labels = batch["labels"].to(device)
    sample_ids = batch["sample_ids"].to(device)

    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key not in {"labels", "sample_ids"}
    }
    return model_inputs, labels, sample_ids


def build_class_weights(
    dataset: FraudMulticlassDataset,
    device: torch.device,
    mode: str,
) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = dataset.class_counts()
    missing = [label_id for label_id, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"训练集中缺少类别 {missing}，无法完成八分类训练。"
        )

    total = sum(counts.values())
    weights = [
        total / (NUM_LABELS * counts[label_id])
        for label_id in range(NUM_LABELS)
    ]

    # 归一化到均值为 1，便于控制损失尺度。
    mean_weight = float(np.mean(weights))
    weights = [weight / mean_weight for weight in weights]

    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    all_labels = list(range(NUM_LABELS))
    target_names = [ID2LABEL[label_id] for label_id in all_labels]

    accuracy = accuracy_score(labels, predictions)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=all_labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=all_labels,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        labels,
        predictions,
        labels=all_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=all_labels,
    )

    # 额外给出“正常/诈骗”折叠后的二分类指标，便于和原实验比较。
    binary_labels = (labels != 0).astype(int)
    binary_predictions = (predictions != 0).astype(int)
    binary_precision, binary_recall, binary_f1, _ = (
        precision_recall_fscore_support(
            binary_labels,
            binary_predictions,
            average="binary",
            zero_division=0,
        )
    )

    fraud_as_normal = int(
        sum(matrix[true_label, 0] for true_label in range(1, NUM_LABELS))
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "binary_normal_vs_fraud": {
            "accuracy": float(
                accuracy_score(binary_labels, binary_predictions)
            ),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1": float(binary_f1),
            "fraud_as_normal_count": fraud_as_normal,
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


@torch.no_grad()
def evaluate_model(
    model: MulticlassCCRClassifier,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    labels_all: list[int] = []
    predictions_all: list[int] = []
    sample_ids_all: list[int] = []
    probabilities_all: list[np.ndarray] = []

    for batch in tqdm(loader, desc="评估", leave=False):
        model_inputs, labels, sample_ids = move_model_inputs(batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits, _ = model(**model_inputs)
            loss = F.cross_entropy(logits, labels, reduction="sum")

        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

        total_loss += float(loss.item())
        total_examples += labels.size(0)

        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        sample_ids_all.extend(sample_ids.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().numpy())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    predictions_np = np.asarray(predictions_all, dtype=np.int64)
    probabilities_np = np.asarray(probabilities_all, dtype=np.float32)

    metrics = compute_metrics(labels_np, predictions_np)
    metrics["loss"] = total_loss / max(total_examples, 1)

    return {
        "metrics": metrics,
        "labels": labels_np,
        "predictions": predictions_np,
        "sample_ids": np.asarray(sample_ids_all, dtype=np.int64),
        "probabilities": probabilities_np,
    }


def predictions_dataframe(
    dataset: FraudMulticlassDataset,
    result: dict[str, Any],
) -> pd.DataFrame:
    frame = dataset.df.reset_index(drop=True).copy()

    if len(frame) != len(result["predictions"]):
        raise RuntimeError(
            "预测数量与测试集数量不一致，请确认 DataLoader 未启用 shuffle。"
        )

    frame["true_label"] = result["labels"]
    frame["true_label_name"] = [
        ID2LABEL[int(label)] for label in result["labels"]
    ]
    frame["prediction"] = result["predictions"]
    frame["prediction_name"] = [
        ID2LABEL[int(label)] for label in result["predictions"]
    ]
    frame["correct"] = (
        result["labels"] == result["predictions"]
    )
    frame["confidence"] = result["probabilities"].max(axis=1)

    for label_id in range(NUM_LABELS):
        frame[f"prob_{label_id}_{ID2LABEL[label_id]}"] = (
            result["probabilities"][:, label_id]
        )

    return frame


def save_checkpoint(
    path: Path,
    model: MulticlassCCRClassifier,
    args: argparse.Namespace,
    stage: str,
    epoch: int,
    validation_metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": args.model_name,
        "num_labels": NUM_LABELS,
        "feature_size": args.feature_size,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "stage": stage,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
    }
    torch.save(checkpoint, path)


def create_dataloaders(
    args: argparse.Namespace,
    tokenizer,
) -> tuple[
    FraudMulticlassDataset,
    FraudMulticlassDataset,
    FraudMulticlassDataset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    train_dataset = FraudMulticlassDataset(
        args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = FraudMulticlassDataset(
        args.val_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_dataset = FraudMulticlassDataset(
        args.test_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    collator = MulticlassCCRDataCollator(tokenizer)

    common = {
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **common,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def train_stage1(
    model: MulticlassCCRClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor | None,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
    output_dir: Path,
) -> pd.DataFrame:
    model.unfreeze_all()

    optimizer = AdamW(
        model.parameters(),
        lr=args.stage1_lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(len(train_loader) * args.stage1_epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    best_macro_f1 = -math.inf
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_cov = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Stage 1 Epoch {epoch}/{args.stage1_epochs}",
        )

        for batch in progress:
            model_inputs, labels, _ = move_model_inputs(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, features = model(**model_inputs)
                ce_loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                )
                covariance_loss = model.covariance_regularization(
                    features
                )
                loss = (
                    ce_loss
                    + args.cov_lambda * covariance_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            seen += batch_size
            running_loss += float(loss.item()) * batch_size
            running_ce += float(ce_loss.item()) * batch_size
            running_cov += (
                float(covariance_loss.item()) * batch_size
            )

            progress.set_postfix(
                loss=f"{running_loss / seen:.4f}",
                ce=f"{running_ce / seen:.4f}",
                cov=f"{running_cov / seen:.4f}",
            )

        validation = evaluate_model(
            model,
            val_loader,
            device,
            amp_enabled,
        )
        val_metrics = validation["metrics"]

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "train_ce": running_ce / max(seen, 1),
            "train_cov": running_cov / max(seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)

        print(
            f"[Stage 1][Epoch {epoch}] "
            f"val_acc={val_metrics['accuracy']:.4%}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4%}, "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4%}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                output_dir / "stage1_best.pt",
                model,
                args,
                stage="stage1",
                epoch=epoch,
                validation_metrics=val_metrics,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        output_dir / "stage1_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return history_df


def compute_ipw_table(
    model: MulticlassCCRClassifier,
    train_dataset: FraudMulticlassDataset,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> tuple[pd.DataFrame, dict[int, float]]:
    if not train_dataset.df["sample_id"].is_unique:
        raise ValueError(
            "训练集 sample_id 不唯一，无法按 sample_id 绑定 IPW。"
        )

    loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=MulticlassCCRDataCollator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    result = evaluate_model(
        model,
        loader,
        device,
        amp_enabled,
    )

    labels = result["labels"]
    predictions = result["predictions"]
    sample_ids = result["sample_ids"]
    correct = (labels == predictions).astype(int)

    group_counts = Counter(
        (int(label), int(is_correct))
        for label, is_correct in zip(labels, correct)
    )
    total = len(labels)

    raw_weights = np.asarray(
        [
            total / group_counts[(int(label), int(is_correct))]
            for label, is_correct in zip(labels, correct)
        ],
        dtype=np.float64,
    )

    # 首先归一化均值，再裁剪极端权重，最后再次归一化。
    normalized = raw_weights / raw_weights.mean()
    clipped = np.minimum(normalized, args.max_ipw)
    final_weights = clipped / clipped.mean()

    table = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": labels,
            "true_label_name": [
                ID2LABEL[int(label)] for label in labels
            ],
            "stage1_prediction": predictions,
            "stage1_prediction_name": [
                ID2LABEL[int(label)] for label in predictions
            ],
            "stage1_correct": correct,
            "group_count": [
                group_counts[(int(label), int(is_correct))]
                for label, is_correct in zip(labels, correct)
            ],
            "raw_ipw": raw_weights,
            "ipw": final_weights,
        }
    )

    weight_map = {
        int(sample_id): float(weight)
        for sample_id, weight in zip(sample_ids, final_weights)
    }
    return table, weight_map


def train_stage2(
    model: MulticlassCCRClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weight_map: dict[int, float],
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
    output_dir: Path,
) -> pd.DataFrame:
    model.freeze_for_stage2()

    optimizer = AdamW(
        model.classifier.parameters(),
        lr=args.stage2_lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(len(train_loader) * args.stage2_epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    best_macro_f1 = -math.inf
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.stage2_epochs + 1):
        # classifier 训练；冻结的编码器和 dropout 保持 eval，
        # 确保 Stage 2 使用固定特征表示。
        model.train()
        model.encoder.eval()
        model.dropout.eval()

        running_loss = 0.0
        running_ce = 0.0
        running_causal = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Stage 2 Epoch {epoch}/{args.stage2_epochs}",
        )

        for batch in progress:
            model_inputs, labels, sample_ids = move_model_inputs(
                batch, device
            )

            try:
                ipw = torch.tensor(
                    [
                        weight_map[int(sample_id)]
                        for sample_id in sample_ids.cpu().tolist()
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            except KeyError as exc:
                raise KeyError(
                    f"找不到 sample_id={exc.args[0]} 对应的 IPW。"
                ) from exc

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, features = model(**model_inputs)
                ce_per_sample = F.cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )
                causal_per_sample = (
                    model.counterfactual_causal_loss(
                        features=features,
                        labels=labels,
                        logits=logits,
                        chunk_size=args.counterfactual_chunk_size,
                    )
                )

                combined_per_sample = (
                    ce_per_sample
                    + args.causal_lambda * causal_per_sample
                )
                loss = (ipw * combined_per_sample).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            seen += batch_size
            running_loss += float(loss.item()) * batch_size
            running_ce += (
                float((ipw * ce_per_sample).mean().item())
                * batch_size
            )
            running_causal += (
                float((ipw * causal_per_sample).mean().item())
                * batch_size
            )

            progress.set_postfix(
                loss=f"{running_loss / seen:.4f}",
                ce=f"{running_ce / seen:.4f}",
                causal=f"{running_causal / seen:.4f}",
            )

        validation = evaluate_model(
            model,
            val_loader,
            device,
            amp_enabled,
        )
        val_metrics = validation["metrics"]

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "train_ipw_ce": running_ce / max(seen, 1),
            "train_ipw_causal": running_causal / max(seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)

        print(
            f"[Stage 2][Epoch {epoch}] "
            f"val_acc={val_metrics['accuracy']:.4%}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4%}, "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4%}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                output_dir / "ccr_best.pt",
                model,
                args,
                stage="stage2",
                epoch=epoch,
                validation_metrics=val_metrics,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        output_dir / "stage2_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return history_df


def save_metric_artifacts(
    prefix: str,
    result: dict[str, Any],
    dataset: FraudMulticlassDataset,
    output_dir: Path,
) -> None:
    predictions_dataframe(dataset, result).to_csv(
        output_dir / f"{prefix}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = pd.DataFrame(
        result["metrics"]["classification_report"]
    ).T
    report.to_csv(
        output_dir / f"{prefix}_classification_report.csv",
        encoding="utf-8-sig",
    )

    matrix = pd.DataFrame(
        result["metrics"]["confusion_matrix"],
        index=[f"真实_{ID2LABEL[i]}" for i in range(NUM_LABELS)],
        columns=[f"预测_{ID2LABEL[i]}" for i in range(NUM_LABELS)],
    )
    matrix.to_csv(
        output_dir / f"{prefix}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    amp_enabled = bool(
        args.fp16 and device.type == "cuda"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"设备：{device}")
    print(f"混合精度：{amp_enabled}")
    print(f"输出目录：{output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(args, tokenizer)

    print("\n训练集类别分布：")
    for label_id, count in train_dataset.class_counts().items():
        print(f"  {label_id} {ID2LABEL[label_id]}：{count}")

    class_weights = build_class_weights(
        train_dataset,
        device,
        args.class_weight_mode,
    )
    if class_weights is not None:
        print(
            "Stage 1 类别权重：",
            [round(float(x), 4) for x in class_weights.cpu()],
        )

    config = {
        **vars(args),
        "num_labels": NUM_LABELS,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "device_used": str(device),
        "amp_enabled": amp_enabled,
        "train_class_counts": train_dataset.class_counts(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = MulticlassCCRClassifier(
        model_name=args.model_name,
        num_labels=NUM_LABELS,
        feature_size=args.feature_size,
        dropout=args.dropout,
    ).to(device)

    print(
        f"\n模型总参数：{sum(p.numel() for p in model.parameters()):,}"
    )

    # -----------------------------
    # Stage 1
    # -----------------------------
    train_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        args=args,
        amp_enabled=amp_enabled,
        output_dir=output_dir,
    )

    stage1_checkpoint = safe_torch_load(
        output_dir / "stage1_best.pt"
    )
    model.load_state_dict(
        stage1_checkpoint["model_state_dict"]
    )

    stage1_test = evaluate_model(
        model,
        test_loader,
        device,
        amp_enabled,
    )
    save_metric_artifacts(
        "stage1_test",
        stage1_test,
        test_dataset,
        output_dir,
    )

    # 根据 Stage 1 对训练集的表现计算 IPW。
    ipw_table, weight_map = compute_ipw_table(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        device=device,
        args=args,
        amp_enabled=amp_enabled,
    )
    ipw_table.to_csv(
        output_dir / "stage2_ipw_train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nIPW 分组统计：")
    print(
        ipw_table.groupby(
            ["true_label_name", "stage1_correct"]
        ).agg(
            samples=("sample_id", "count"),
            mean_ipw=("ipw", "mean"),
            max_ipw=("ipw", "max"),
        ).to_string()
    )

    # -----------------------------
    # Stage 2
    # -----------------------------
    train_stage2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        weight_map=weight_map,
        device=device,
        args=args,
        amp_enabled=amp_enabled,
        output_dir=output_dir,
    )

    final_checkpoint = safe_torch_load(
        output_dir / "ccr_best.pt"
    )
    model.load_state_dict(
        final_checkpoint["model_state_dict"]
    )

    final_test = evaluate_model(
        model,
        test_loader,
        device,
        amp_enabled,
    )
    save_metric_artifacts(
        "ccr_test",
        final_test,
        test_dataset,
        output_dir,
    )

    test_metrics = {
        "stage1_test": stage1_test["metrics"],
        "final_ccr_test": final_test["metrics"],
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("八分类 CCR 训练与测试完成")
    print("=" * 80)

    for name, result in [
        ("Stage 1", stage1_test),
        ("Final CCR", final_test),
    ]:
        metrics = result["metrics"]
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']:.4%}, "
            f"Macro-F1={metrics['macro_f1']:.4%}, "
            f"Weighted-F1={metrics['weighted_f1']:.4%}, "
            f"诈骗误判为正常="
            f"{metrics['binary_normal_vs_fraud']['fraud_as_normal_count']}"
        )

    print(f"\n最佳 Stage 1：{output_dir / 'stage1_best.pt'}")
    print(f"最终 CCR：{output_dir / 'ccr_best.pt'}")
    print(f"完整指标：{output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
