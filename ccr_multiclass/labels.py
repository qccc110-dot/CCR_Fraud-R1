from __future__ import annotations

NUM_LABELS = 8

LABEL2ID = {
    "正常通话": 0,
    "客服诈骗": 1,
    "银行诈骗": 2,
    "投资诈骗": 3,
    "钓鱼诈骗": 4,
    "彩票诈骗": 5,
    "绑架诈骗": 6,
    "身份盗窃": 7,
}

ID2LABEL = {label_id: label_name for label_name, label_id in LABEL2ID.items()}
