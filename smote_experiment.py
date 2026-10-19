import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读数据
df = pd.read_csv("data/students.csv").fillna(pd.read_csv("data/students.csv").mean(numeric_only=True))
features = ["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support"]
X = df[features]
y = df["at_risk"]

# 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"原始训练集：正样本 {sum(y_train == 1)} 个，负样本 {sum(y_train == 0)} 个")

# 2. 应用 SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE后训练集：正样本 {sum(y_train_smote == 1)} 个，负样本 {sum(y_train_smote == 0)} 个")

# 3. 训练并对比
models = {
    "原始随机森林": RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42).fit(X_train, y_train),
    "SMOTE随机森林": RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42).fit(X_train_smote,
                                                                                                y_train_smote)
}

plt.figure(figsize=(8, 6))
results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results.append([name, acc, rec, auc])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("假阳性率 FPR")
plt.ylabel("真阳性率 TPR")
plt.title("SMOTE处理前后 ROC曲线对比")
plt.legend()
plt.savefig("output/smote_roc.png", dpi=150)
plt.close()

print("\n=== SMOTE 处理前后指标对比 ===")
res_df = pd.DataFrame(results, columns=["模型", "准确率", "召回率", "AUC"]).round(4)
print(res_df.to_string(index=False))
print("\n图表已保存至 output/smote_roc.png")