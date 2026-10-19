import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 设置中文字体，防止图表方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读取并预处理数据
df = pd.read_csv("data/students.csv").fillna(pd.read_csv("data/students.csv").mean(numeric_only=True))
features = ["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support"]
X = df[features]
y = df["at_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ================= 实验 1：决策树深度 =================
depths = range(2, 16)
train_acc, test_acc = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dt.predict(X_train)))
    test_acc.append(accuracy_score(y_test, dt.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_acc, marker='o', label="训练集准确率")
plt.plot(depths, test_acc, marker='s', label="测试集准确率")
plt.xlabel("决策树最大深度")
plt.ylabel("准确率")
plt.title("决策树深度对准确率的影响")
plt.legend()
plt.savefig("output/exp1_tree_depth.png", dpi=150)
plt.close()

# ================= 实验 2：随机森林树数量 =================
n_estimators = [10, 30, 50, 80, 100, 150, 200]
rf_auc = []
for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train, y_train)
    rf_auc.append(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

plt.figure(figsize=(8, 5))
plt.plot(n_estimators, rf_auc, marker='^', color='green', linewidth=2)
plt.xlabel("随机森林树的数量")
plt.ylabel("AUC")
plt.title("树数量对AUC的影响")
plt.savefig("output/exp2_rf_estimators.png", dpi=150)
plt.close()

# ================= 实验 3：SHAP 可解释性 =================
# 训练一个基础模型用于解释
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# 计算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 图1：SHAP 蜂群图
plt.figure()
shap.summary_plot(shap_values[1], X, show=False)
plt.savefig("output/shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()

# 图2：SHAP 单学生瀑布图 (适配当前 shap 版本)
probas = model.predict_proba(X)[:, 1]
high_risk_idx = np.argmax(probas)

# 构造 Explanation 对象用于画图
explanation = explainer(X.iloc[[high_risk_idx]])
plt.figure()
shap.plots.waterfall(explanation[0, :, 1], show=False)
plt.savefig("output/shap_waterfall.png", dpi=150, bbox_inches='tight')
plt.close()

print("全部图表（含 SHAP）已成功生成到 output 文件夹！")