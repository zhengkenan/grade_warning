import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
df = pd.read_csv("data/students.csv").fillna(0)

# 计算相关系数并画图
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特征相关性热力图")
plt.savefig("output/eda_heatmap.png", dpi=150)
plt.close()
print("热力图已保存至 output/eda_heatmap.png")