from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "诈骗", "是"}:
        return 1
    if s in {"false", "0", "no", "n", "非诈骗", "否", "正常"}:
        return 0
    return None


def clean_text(value: str) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_dataframe(path: Path, source_split: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = {"specific_dialogue_content", "is_fraud"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")

    before = len(raw)
    raw["text"] = raw["specific_dialogue_content"].map(clean_text)
    raw["label"] = raw["is_fraud"].map(parse_label)
    invalid_label = int(raw["label"].isna().sum())
    empty_text = int(raw["text"].eq("").sum())

    keep_meta = [c for c in ["interaction_strategy", "call_type", "fraud_type"] if c in raw.columns]
    df = raw[["text", "label", *keep_meta]].copy()
    df = df[df["label"].notna() & df["text"].ne("")].copy()
    df["label"] = df["label"].astype(int)
    df["source_split"] = source_split

    duplicated = int(df.duplicated(subset=["text", "label"]).sum())
    df = df.drop_duplicates(subset=["text", "label"], keep="first").reset_index(drop=True)

    report = {
        "input_file": str(path),
        "rows_before": before,
        "invalid_or_missing_label_rows": invalid_label,
        "empty_text_rows": empty_text,
        "duplicate_text_label_rows_removed": duplicated,
        "rows_after": len(df),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
    }
    return df, report


def main():
    parser = argparse.ArgumentParser(description="将原始课程 CSV 整理成 CCR 可训练格式")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--test-input", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_all, train_report = clean_dataframe(Path(args.train_input), "provided_train")
    test_df, test_report = clean_dataframe(Path(args.test_input), "provided_test")

    overlap = set(train_all["text"]) & set(test_df["text"])
    if overlap:
        train_all = train_all[~train_all["text"].isin(overlap)].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_all,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_all["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df = df.copy()
        df.insert(0, "sample_id", range(len(df)))
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    report = {
        "train_cleaning": train_report,
        "test_cleaning": test_report,
        "exact_text_overlap_removed_from_train": len(overlap),
        "final_splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "label_meanings": {"0": "非诈骗", "1": "诈骗"},
    }
    (out / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
