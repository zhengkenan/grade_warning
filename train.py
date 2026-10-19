import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

plt.rcParams["font.sans-serif"] = ["SimHei"]   # 中文显示
plt.rcParams["axes.unicode_minus"] = False

# 1. 读数据
df = pd.read_csv("data/students.csv")

# 2. 缺失值处理（均值填充）
df = df.fillna(df.mean(numeric_only=True))

# 3. 特征与标签
features = ["attendance", "homework_rate", "study_hours",
            "past_fail", "online_hours", "family_support"]
X = df[features]
y = df["at_risk"]

# 4. 划分训练/测试集（8:2，分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. 标准化（逻辑回归需要，树模型不影响）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 6. 三个模型
models = {
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "决策树": DecisionTreeClassifier(max_depth=5, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
plt.figure(figsize=(7, 6))

for name, model in models.items():
    if name == "逻辑回归":
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results.append([name, acc, rec, f1, auc])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="随机猜测")
plt.xlabel("假阳性率 FPR")
plt.ylabel("真阳性率 TPR")
plt.title("三种模型 ROC 曲线对比")
plt.legend()
plt.tight_layout()
plt.savefig("output/roc_curve.png", dpi=150)
plt.close()

# 7. 指标表格
res_df = pd.DataFrame(results, columns=["模型", "准确率", "召回率", "F1", "AUC"])
res_df = res_df.round(4)
res_df.to_csv("output/model_metrics.csv", index=False, encoding="utf-8-sig")
print("\n=== 模型指标对比 ===")
print(res_df.to_string(index=False))

# 8. 随机森林特征重要性
rf = models["随机森林"]
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
imp.plot(kind="bar", color="steelblue")
plt.title("随机森林特征重要性排序")
plt.ylabel("重要性")
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150)
plt.close()

print("\n=== 特征重要性 ===")
print(imp.round(4).to_string())

# 9. 最优模型的混淆矩阵
best_name = res_df.loc[res_df["AUC"].idxmax(), "模型"]
best_model = models[best_name]
if best_name == "逻辑回归":
    y_pred_best = best_model.predict(X_test_s)
else:
    y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n=== 最优模型（{best_name}）混淆矩阵 ===")
print(cm)
print("\n图表已保存到 output/ 目录")