from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .labels import NUM_LABELS


class MulticlassCCRClassifier(nn.Module):
    """面向中文八分类任务的 CCR 模型。

    八个类别：
      0 正常通话
      1 客服诈骗
      2 银行诈骗
      3 投资诈骗
      4 钓鱼诈骗
      5 彩票诈骗
      6 绑架诈骗
      7 身份盗窃

    网络结构：
      Transformer pooled representation
        -> dropout
        -> feature layer
        -> tanh
        -> 8-class classifier

    Stage 1:
      多分类交叉熵 + 协方差非对角项正则

    Stage 2:
      冻结 encoder 和 feature layer，仅训练 classifier；
      使用“真实类别 × Stage 1 是否预测正确”计算的 IPW；
      加入逐特征反事实遮蔽的因果约束。
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        feature_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_labels < 2:
            raise ValueError(f"num_labels 必须至少为 2，当前为 {num_labels}")

        self.model_name = model_name
        self.num_labels = num_labels
        self.feature_size = feature_size

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.feature_layer = nn.Linear(hidden_size, feature_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(feature_size, num_labels)

    def encode_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)
        features = self.activation(self.feature_layer(pooled))
        return features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.classifier(features)
        return logits, features

    @staticmethod
    def covariance_regularization(features: torch.Tensor) -> torch.Tensor:
        """计算特征协方差矩阵非对角项的 Frobenius 范数。"""
        if features.size(0) <= 1:
            return features.new_zeros(())

        centered = features - features.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (features.size(0) - 1)

        diagonal = torch.diagonal(covariance)
        off_diagonal = covariance - torch.diag_embed(diagonal)
        return torch.norm(off_diagonal, p="fro")

    def counterfactual_causal_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """多分类版本的 CCR 反事实因果损失。

        对每个特征维度逐一置零，并比较“真实类别”的原始概率和
        反事实概率。这里不能固定读取第 1 类概率，而必须根据每条
        样本的 labels 使用 gather 提取其真实类别概率。

        返回 shape=[batch_size] 的逐样本损失，便于后续乘 IPW。
        """
        if labels.dtype != torch.long:
            labels = labels.long()

        if labels.numel() > 0:
            min_label = int(labels.min().item())
            max_label = int(labels.max().item())
            if min_label < 0 or max_label >= self.num_labels:
                raise ValueError(
                    f"标签范围应为 [0, {self.num_labels - 1}]，"
                    f"当前最小值={min_label}，最大值={max_label}"
                )

        batch_size, feature_size = features.shape

        # 原始真实类别概率只作为比较基准，不需要回传梯度。
        with torch.no_grad():
            raw_probabilities = F.softmax(logits, dim=-1)
            p_raw = raw_probabilities.gather(
                1, labels.unsqueeze(1)
            ).squeeze(1)

        log_terms = []

        for start in range(0, feature_size, chunk_size):
            end = min(start + chunk_size, feature_size)
            dimensions = torch.arange(
                start, end, device=features.device
            )
            width = end - start

            # [B, width, D]：每个副本遮蔽一个不同特征维度。
            counterfactual_features = (
                features[:, None, :]
                .expand(batch_size, width, feature_size)
                .clone()
            )

            local_columns = torch.arange(
                width, device=features.device
            )
            counterfactual_features[
                :, local_columns, dimensions
            ] = 0.0

            counterfactual_logits = self.classifier(
                counterfactual_features.reshape(-1, feature_size)
            )
            counterfactual_probabilities = F.softmax(
                counterfactual_logits, dim=-1
            )

            repeated_labels = (
                labels[:, None]
                .expand(batch_size, width)
                .reshape(-1)
            )

            p_counterfactual = counterfactual_probabilities.gather(
                1, repeated_labels.unsqueeze(1)
            ).reshape(batch_size, width)

            z = p_raw[:, None] - p_counterfactual + 1.0
            z = torch.clamp(z, min=1.0)
            log_terms.append(torch.log(z))

        log_z = torch.cat(log_terms, dim=1)
        return -log_z.mean(dim=1)

    def freeze_for_stage2(self) -> None:
        """Stage 2：冻结特征提取器，只训练八分类头。"""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for parameter in self.feature_layer.parameters():
            parameter.requires_grad = False

        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_all(self) -> None:
        """恢复全部参数可训练，用于 Stage 1。"""
        for parameter in self.parameters():
            parameter.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
