from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CCRClassifier(nn.Module):
    """面向中文二分类任务的紧凑 CCR 实现。

    结构与论文/官方代码一致：
      Transformer pooled representation -> feature layer -> tanh -> classifier

    Stage 1:
      CE + 协方差非对角项正则（特征解耦）

    Stage 2:
      冻结 encoder 和 feature layer，仅训练 classifier；
      使用按“类别 × 首阶段是否误分类”估计的 IPW；
      加入逐特征反事实遮蔽的因果约束。
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        feature_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.feature_size = feature_size
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.feature_layer = nn.Linear(hidden_size, feature_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(feature_size, num_labels)

    def encode_features(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.activation(self.feature_layer(pooled))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        features = self.encode_features(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(features)
        return logits, features

    @staticmethod
    def covariance_regularization(features: torch.Tensor) -> torch.Tensor:
        """官方实现使用协方差矩阵非对角部分的 Frobenius 范数。"""
        if features.size(0) <= 1:
            return features.new_zeros(())
        centered = features - features.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (features.size(0) - 1)
        off_diag = covariance - torch.diag_embed(torch.diagonal(covariance))
        return torch.norm(off_diag, p="fro")

    def counterfactual_causal_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """官方 CCR counterfact 逻辑的省显存版本。

        对每个特征维度逐一置零，比较真实标签的原始概率与反事实概率：
            z = p_raw(y) - p_cf(y) + 1
            z = max(z, 1)
            L_causal = -mean_j log(z_j)

        返回每个样本一个损失，便于随后乘 IPW。
        """
        batch_size, feature_size = features.shape
        with torch.no_grad():
            p_raw = F.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)

        losses = []
        for start in range(0, feature_size, chunk_size):
            end = min(start + chunk_size, feature_size)
            dims = torch.arange(start, end, device=features.device)
            width = end - start

            cf = features[:, None, :].expand(batch_size, width, feature_size).clone()
            row = torch.arange(width, device=features.device)
            cf[:, row, dims] = 0.0
            cf_logits = self.classifier(cf.reshape(-1, feature_size))
            cf_probs = F.softmax(cf_logits, dim=-1)
            repeated_labels = labels[:, None].expand(batch_size, width).reshape(-1)
            p_cf = cf_probs.gather(1, repeated_labels[:, None]).reshape(batch_size, width)

            z = p_raw[:, None] - p_cf + 1.0
            z = torch.clamp(z, min=1.0)
            losses.append(torch.log(z))

        log_z = torch.cat(losses, dim=1)
        return -log_z.mean(dim=1)

    def freeze_for_stage2(self) -> None:
        """论文 Stage 2：固定特征提取器，只训练最后分类层。"""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.feature_layer.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
