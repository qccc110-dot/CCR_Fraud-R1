from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import ID2LABEL, NUM_LABELS


REQUIRED_COLUMNS = ["text", "label"]


class FraudMulticlassDataset(Dataset):
    """CCR 八分类数据集。

    CSV 至少包含：
      - text：对话文本
      - label：0~7 的八分类标签

    推荐同时包含：
      - label_name：中文类别名称
      - binary_label：原二分类标签
      - sample_id：样本编号

    其他字段仅作为元数据保存在 self.df 中，不输入模型。
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        max_length: int = 384,
        num_labels: int = NUM_LABELS,
        validate_label_name: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(
            self.csv_path,
            encoding="utf-8-sig",
        )

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in self.df.columns
        ]
        if missing:
            raise ValueError(
                f"{self.csv_path} 缺少必要字段：{missing}"
            )

        if self.df["text"].isna().any():
            count = int(self.df["text"].isna().sum())
            raise ValueError(
                f"{self.csv_path} 中存在 {count} 条空文本"
            )

        self.df["label"] = self.df["label"].astype(int)

        invalid_mask = ~self.df["label"].between(
            0, num_labels - 1
        )
        if invalid_mask.any():
            invalid_values = sorted(
                self.df.loc[invalid_mask, "label"]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"{self.csv_path} 含有超出范围的标签："
                f"{invalid_values}；合法范围为 0~{num_labels - 1}"
            )

        if (
            validate_label_name
            and "label_name" in self.df.columns
        ):
            expected_names = self.df["label"].map(ID2LABEL)
            inconsistent = (
                self.df["label_name"].astype(str)
                != expected_names.astype(str)
            )
            if inconsistent.any():
                preview = self.df.loc[
                    inconsistent,
                    ["sample_id", "label", "label_name"],
                ].head(5)
                raise ValueError(
                    "label 与 label_name 不一致，示例：\n"
                    + preview.to_string(index=False)
                )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Dict[str, torch.Tensor | int]:
        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        encoded["labels"] = int(row["label"])
        encoded["sample_id"] = int(
            row.get("sample_id", idx)
        )
        return encoded

    def class_counts(self) -> dict[int, int]:
        counts = (
            self.df["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        return {
            label_id: int(counts.get(label_id, 0))
            for label_id in range(self.num_labels)
        }


@dataclass
class MulticlassCCRDataCollator:
    tokenizer: object

    def __call__(
        self,
        features: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        labels = torch.tensor(
            [feature.pop("labels") for feature in features],
            dtype=torch.long,
        )
        sample_ids = torch.tensor(
            [
                feature.pop("sample_id")
                for feature in features
            ],
            dtype=torch.long,
        )

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels
        batch["sample_ids"] = sample_ids
        return batch
