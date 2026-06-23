from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ccr.data import CCRDataCollator, FraudTextDataset
from ccr.model import CCRClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs, all_ids = [], [], [], []
    total_loss = 0.0
    total_n = 0
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        sample_ids = batch.pop("sample_ids")
        logits, _ = model(**batch)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        total_loss += loss.item()
        total_n += labels.numel()
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_ids.extend(sample_ids.cpu().tolist())
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics, pd.DataFrame({
        "sample_id": all_ids,
        "label": all_labels,
        "prediction": all_preds,
        "fraud_probability": all_probs,
    })


def make_optimizer_and_scheduler(model, loader, epochs, lr, weight_decay, warmup_ratio):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = max(1, len(loader) * epochs)
    warmup = int(steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps)
    return optimizer, scheduler


def train_stage1(model, train_loader, val_loader, device, args, output_dir):
    model.unfreeze_all()
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, train_loader, args.stage1_epochs, args.stage1_lr,
        args.weight_decay, args.warmup_ratio,
    )
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"Stage1 epoch {epoch}")
        for batch in bar:
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            batch.pop("sample_ids")
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(**batch)
            ce = F.cross_entropy(logits, labels)
            cov = model.covariance_regularization(features)
            loss = ce + args.lambda_cov * cov
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            bar.set_postfix(loss=f"{running/max(seen,1):.4f}")

        val_metrics, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), **val_metrics}
        history.append(row)
        print("Stage1", json.dumps(row, ensure_ascii=False))
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), output_dir / "stage1_best.pt")
    pd.DataFrame(history).to_csv(output_dir / "stage1_history.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(torch.load(output_dir / "stage1_best.pt", map_location=device))


@torch.no_grad()
def estimate_ipw(model, train_loader, device, output_dir):
    """按论文思路：每个类别内，Stage1 正确样本视为多数组，错误样本视为少数组。"""
    model.eval()
    records = []
    for batch in tqdm(train_loader, desc="estimate IPW", leave=False):
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        sample_ids = batch.pop("sample_ids")
        logits, _ = model(**batch)
        preds = logits.argmax(dim=-1)
        correct = preds.eq(labels)
        for sid, y, ok in zip(sample_ids.cpu().tolist(), labels.cpu().tolist(), correct.cpu().tolist()):
            records.append((sid, y, int(ok)))

    df = pd.DataFrame(records, columns=["sample_id", "label", "correct_stage1"])
    counts = df.groupby(["label", "correct_stage1"]).size().to_dict()
    class_counts = df.groupby("label").size().to_dict()

    weights = {}
    eps = 1e-6
    for row in df.itertuples(index=False):
        group_count = counts.get((row.label, row.correct_stage1), 0)
        propensity = group_count / max(class_counts[row.label], 1)
        weights[int(row.sample_id)] = 1.0 / max(2.0 * propensity, eps)

    mean_w = float(np.mean(list(weights.values())))
    weights = {k: v / mean_w for k, v in weights.items()}
    df["ipw"] = df["sample_id"].map(weights)
    df.to_csv(output_dir / "stage2_ipw_train.csv", index=False, encoding="utf-8-sig")
    return weights


def train_stage2(model, train_loader, val_loader, device, args, output_dir, weights):
    model.freeze_for_stage2()
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, train_loader, args.stage2_epochs, args.stage2_lr,
        args.stage2_weight_decay, args.warmup_ratio,
    )
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.stage2_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"Stage2 epoch {epoch}")
        for batch in bar:
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            sample_ids = batch.pop("sample_ids")
            sample_weights = torch.tensor(
                [weights[int(i)] for i in sample_ids.cpu().tolist()],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(**batch)
            ce_each = F.cross_entropy(logits, labels, reduction="none")
            causal_each = model.counterfactual_causal_loss(
                features, labels, logits, chunk_size=args.counterfactual_chunk_size
            )
            loss_each = ce_each + args.lambda_causal * causal_each
            loss = (loss_each * sample_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            bar.set_postfix(loss=f"{running/max(seen,1):.4f}")

        val_metrics, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), **val_metrics}
        history.append(row)
        print("Stage2", json.dumps(row, ensure_ascii=False))
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), output_dir / "ccr_best.pt")
    pd.DataFrame(history).to_csv(output_dir / "stage2_history.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(torch.load(output_dir / "ccr_best.pt", map_location=device))


def main():
    p = argparse.ArgumentParser(description="训练中文 CCR 虚假通话二分类模型")
    p.add_argument("--train-file", default="data/processed/train.csv")
    p.add_argument("--val-file", default="data/processed/val.csv")
    p.add_argument("--test-file", default="data/processed/test.csv")
    p.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--output-dir", default="outputs/ccr_run")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--feature-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--stage1-epochs", type=int, default=3)
    p.add_argument("--stage2-epochs", type=int, default=10)
    p.add_argument("--stage1-lr", type=float, default=2e-5)
    p.add_argument("--stage2-lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--stage2-weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-cov", type=float, default=0.5)
    p.add_argument("--lambda-causal", type=float, default=0.1)
    p.add_argument("--counterfactual-chunk-size", type=int, default=32)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    collator = CCRDataCollator(tokenizer)
    train_ds = FraudTextDataset(args.train_file, tokenizer, args.max_length)
    val_ds = FraudTextDataset(args.val_file, tokenizer, args.max_length)
    test_ds = FraudTextDataset(args.test_file, tokenizer, args.max_length)

    common = dict(collate_fn=collator, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    train_eval_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, **common)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, **common)

    model = CCRClassifier(args.model_name, num_labels=2, feature_size=args.feature_size).to(device)

    print("\n========== Stage 1: ERM + covariance disentanglement ==========")
    train_stage1(model, train_loader, val_loader, device, args, output_dir)
    stage1_test_metrics, stage1_preds = evaluate(model, test_loader, device)
    stage1_preds.to_csv(output_dir / "stage1_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n========== Estimate IPW from Stage 1 errors ==========")
    ipw = estimate_ipw(model, train_eval_loader, device, output_dir)

    print("\n========== Stage 2: classifier-only + IPW + causal constraint ==========")
    train_stage2(model, train_loader, val_loader, device, args, output_dir, ipw)
    ccr_test_metrics, ccr_preds = evaluate(model, test_loader, device)
    ccr_preds.to_csv(output_dir / "ccr_test_predictions.csv", index=False, encoding="utf-8-sig")

    metrics = {"stage1_test": stage1_test_metrics, "final_ccr_test": ccr_test_metrics}
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    print("\n最终结果：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"模型与结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
