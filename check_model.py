
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, recall_score

# 1. 读取数据
df = pd.read_csv("data/students.csv")

features = [
    "attendance",
    "homework_rate",
    "study_hours",
    "past_fail",
    "online_hours",
    "family_support"
]
target = "at_risk"

# 检查必要列
missing_cols = [c for c in features + [target] if c not in df.columns]
if missing_cols:
    raise ValueError(f"数据缺少必要字段：{missing_cols}")

X = df[features].copy()
y = df[target].copy()

if y.isna().any():
    raise ValueError("at_risk 标签存在缺失值，请先检查标签生成过程。")

if set(y.unique()) != {0, 1}:
    raise ValueError(f"at_risk 应为 0/1 标签，当前值为：{sorted(y.unique())}")

print("=" * 50)
print("一、数据基本情况")
print("=" * 50)
print(f"样本数量：{len(df)}")
print(f"输入特征：{features}")
print(f"特征缺失值数量：\n{X.isna().sum()}")
print(f"标签数量分布：\n{y.value_counts().sort_index()}")
print(f"标签比例：\n{y.value_counts(normalize=True).sort_index().round(4)}")

# 2. 检查可能的目标泄漏字段
print("\n" + "=" * 50)
print("二、字段检查")
print("=" * 50)

suspicious_names = [
    "final_score", "at_risk", "risk_score",
    "risk_level", "prediction", "predicted_risk"
]
found = [
    c for c in df.columns
    if c.lower() in suspicious_names and c not in features
]
print("数据中可能与标签或预测结果直接相关的字段：", found)
print("注意：student_id 等标识字段也不应直接作为模型输入。")

# 3. 查看特征与标签的关系
print("\n" + "=" * 50)
print("三、特征与标签的关联")
print("=" * 50)

corr_df = X.copy()
corr_df[target] = y
print("特征与标签的相关系数（仅作初步参考）：")
print(corr_df.corr(numeric_only=True)[target].drop(target).round(4))

print("\n各特征在两类学生中的均值：")
print(df.groupby(target)[features].mean().T.round(3))

# 4. 建立模型：预处理全部放在 Pipeline 内
models = {
    "逻辑回归": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "决策树": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", DecisionTreeClassifier(
            max_depth=5, random_state=42
        ))
    ]),
    "随机森林": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42
        ))
    ])
}

# 5. 5 折分层交叉验证
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "auc": "roc_auc",
    "accuracy": "accuracy",
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0)
}

print("\n" + "=" * 50)
print("四、5 折分层交叉验证")
print("=" * 50)

all_results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    auc_scores = scores["test_auc"]
    acc_scores = scores["test_accuracy"]
    recall_scores = scores["test_recall"]
    f1_scores = scores["test_f1"]

    print(f"\n【{name}】")
    print("每折 AUC：", np.round(auc_scores, 4))
    print(f"AUC 均值 ± 标准差：{auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    print(f"准确率均值：{acc_scores.mean():.4f}")
    print(f"召回率均值：{recall_scores.mean():.4f}")
    print(f"F1 均值：{f1_scores.mean():.4f}")

    all_results.append({
        "模型": name,
        "AUC均值": auc_scores.mean(),
        "AUC标准差": auc_scores.std(),
        "准确率均值": acc_scores.mean(),
        "召回率均值": recall_scores.mean(),
        "F1均值": f1_scores.mean()
    })

result_df = pd.DataFrame(all_results).round(4)
os.makedirs("output", exist_ok=True)
result_df.to_csv(
    "output/cross_validation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 50)
print("五、模型汇总")
print("=" * 50)
print(result_df.to_string(index=False))
print("\n交叉验证结果已保存到 output/cross_validation_results.csv")
