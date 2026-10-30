
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42
DATA_PATH = "data/students.csv"
OUTPUT_DIR = "output"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读取并检查数据
df = pd.read_csv(DATA_PATH)

features = [
    "attendance",
    "homework_rate",
    "study_hours",
    "past_fail",
    "online_hours",
    "family_support"
]
target = "at_risk"

required_columns = features + [target]
missing_columns = [
    col for col in required_columns if col not in df.columns
]
if missing_columns:
    raise ValueError(f"数据缺少必要字段：{missing_columns}")

df = df[required_columns].copy()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[target].isna().any():
    raise ValueError("风险标签存在缺失或非数字值，请先检查数据。")

if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError("at_risk 必须使用 0 和 1 表示。")

if df[target].nunique() != 2:
    raise ValueError("当前实验要求数据中同时存在 0 和 1 两类标签。")

X = df[features]
y = df[target].astype(int)

if X.isna().all(axis=0).any():
    bad_cols = X.columns[X.isna().all()].tolist()
    raise ValueError(f"以下特征全部缺失，无法进行均值填补：{bad_cols}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"数据总量：{len(df)}")
print(f"训练集：{len(X_train)}")
print(f"测试集：{len(X_test)}")
print(f"风险样本比例：{y.mean():.2%}")

# 3. 建立统一的预处理与模型流程
models = {
    "逻辑回归": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ]),
    "决策树": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        ))
    ]),
    "随机森林": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}

# 4. 仅使用训练集进行分层交叉验证
class_counts = y_train.value_counts()
n_splits = min(5, int(class_counts.min()))

if n_splits < 2:
    raise ValueError("训练集中某一类别样本不足，无法进行分层交叉验证。")

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "auc": "roc_auc"
}

cv_results = []
test_results = []
fitted_models = {}
test_probabilities = {}

for name, pipeline in models.items():
    print(f"\n正在评估：{name}")

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise"
    )

    cv_results.append({
        "模型": name,
        "CV准确率均值": scores["test_accuracy"].mean(),
        "CV准确率标准差": scores["test_accuracy"].std(),
        "CV精确率均值": scores["test_precision"].mean(),
        "CV召回率均值": scores["test_recall"].mean(),
        "CV_F1均值": scores["test_f1"].mean(),
        "CV_AUC均值": scores["test_auc"].mean(),
        "CV_AUC标准差": scores["test_auc"].std()
    })

    # 交叉验证结束后，在完整训练集上拟合
    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)

    y_pred = fitted.predict(X_test)
    y_prob = fitted.predict_proba(X_test)[:, 1]

    test_results.append({
        "模型": name,
        "测试集准确率": accuracy_score(y_test, y_pred),
        "测试集精确率": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "测试集召回率": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "测试集F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "测试集AUC": roc_auc_score(y_test, y_prob)
    })

    fitted_models[name] = fitted
    test_probabilities[name] = y_prob

# 5. 保存交叉验证与测试集指标
cv_df = pd.DataFrame(cv_results).round(4)
test_df = pd.DataFrame(test_results).round(4)

cv_df.to_csv(
    f"{OUTPUT_DIR}/cv_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)
test_df.to_csv(
    f"{OUTPUT_DIR}/model_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 交叉验证结果 ===")
print(cv_df.to_string(index=False))

print("\n=== 独立测试集结果 ===")
print(test_df.to_string(index=False))

# 6. 绘制 ROC 曲线
plt.figure(figsize=(7, 6))

for name, y_prob in test_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="随机分类参考线")
plt.xlabel("假阳性率 FPR")
plt.ylabel("真阳性率 TPR")
plt.title("三种模型 ROC 曲线对比")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
plt.close()

# 7. 按交叉验证 AUC 选择候选模型
# 测试集不参与模型选择，避免测试集信息泄漏
best_name = max(
    cv_results,
    key=lambda item: item["CV_AUC均值"]
)["模型"]

best_model = fitted_models[best_name]
best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["非风险", "风险"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title(f"{best_name}：测试集混淆矩阵")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# 8. 保存候选模型和实验元数据
joblib.dump(
    best_model,
    f"{MODEL_DIR}/best_model.joblib"
)

metadata = {
    "selected_model": best_name,
    "features": features,
    "target": target,
    "random_state": RANDOM_STATE,
    "test_size": 0.2,
    "cv_splits": n_splits,
    "selection_metric": "CV_AUC均值",
    "positive_label": 1
}

with open(
    f"{MODEL_DIR}/metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

# 9. 随机森林特征重要性
rf_pipeline = fitted_models["随机森林"]
rf_model = rf_pipeline.named_steps["model"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

importance.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    header=["重要性"],
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.xlabel("特征重要性")
plt.title("随机森林特征重要性排序")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
plt.close()

print(f"\n交叉验证 AUC 最优候选模型：{best_name}")
print("模型已保存至 models/best_model.joblib")
print("指标、混淆矩阵和图表已保存至 output/ 目录")
