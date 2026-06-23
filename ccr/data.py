from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ["text", "label"]


class FraudTextDataset(Dataset):
    """CCR 二分类数据集。

    CSV 至少包含：
      - text: 对话文本
      - label: 0=非诈骗，1=诈骗
    其余列会作为元数据保留，但不输入模型。
    """

    def __init__(self, csv_path: str | Path, tokenizer, max_length: int = 384):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"{self.csv_path} 缺少必要字段: {missing}")
        self.df["label"] = self.df["label"].astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        encoded["labels"] = int(row["label"])
        encoded["sample_id"] = int(row.get("sample_id", idx))
        return encoded


@dataclass
class CCRDataCollator:
    tokenizer: object

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
        sample_ids = torch.tensor([f.pop("sample_id") for f in features], dtype=torch.long)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["sample_ids"] = sample_ids
        return batch
