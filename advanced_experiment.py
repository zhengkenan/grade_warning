import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/students.csv").fillna(pd.read_csv("data/students.csv").mean(numeric_only=True))
features = ["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support"]
X = StandardScaler().fit_transform(df[features])
y = df["at_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(">>> 正在执行网格搜索...")
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [5, 8, 10], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"最优参数: {grid_search.best_params_}")

print(">>> 正在训练融合模型...")
lr = LogisticRegression(max_iter=1000, random_state=42)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
voting_clf = VotingClassifier(estimators=[('lr', lr), ('dt', dt), ('rf', best_rf)], voting='soft').fit(X_train, y_train)

models = {"最优随机森林": best_rf, "Voting融合模型": voting_clf}
plt.figure(figsize=(8, 6))
results = []
for name, model in models.items():
    y_pred, y_prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    results.append([name, accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_prob)])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("最优随机森林 vs Voting融合模型")
plt.legend()
plt.savefig("output/advanced_roc.png", dpi=150)
plt.close()

print("\n=== 高级模型指标对比 ===")
print(pd.DataFrame(results, columns=["模型", "准确率", "召回率", "F1", "AUC"]).round(4).to_string(index=False))