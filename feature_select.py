import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 读数据
df = pd.read_csv("data/students.csv").fillna(0)
y = df["at_risk"]

# 方案A：用全部6个特征
X_all = df[["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support"]]
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
auc_all = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"全部特征 - AUC: {auc_all:.4f}")

# 方案B：只用最重要的3个特征（挂科数、出勤率、作业提交率）
X_top3 = df[["past_fail", "attendance", "homework_rate"]]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_top3, y, test_size=0.2, random_state=42, stratify=y)
rf3 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train3, y_train3)
auc_top3 = roc_auc_score(y_test3, rf3.predict_proba(X_test3)[:, 1])
print(f"Top3特征 - AUC: {auc_top3:.4f}")