import pandas as pd
import numpy as np

np.random.seed(42)
N = 1000

# 基础特征（模拟真实分布）
attendance = np.clip(np.random.normal(0.85, 0.12, N), 0.3, 1.0)          # 出勤率
homework_rate = np.clip(np.random.normal(0.80, 0.15, N), 0.2, 1.0)       # 作业提交率
study_hours = np.clip(np.random.normal(8, 3, N), 1, 20)                  # 每周学习时长
past_fail = np.random.poisson(0.8, N)                                    # 过往挂科数
online_hours = np.clip(np.random.normal(5, 2, N), 0, 15)                 # 在线学习时长
family_support = np.random.choice([0, 1], N, p=[0.35, 0.65])             # 家庭支持 0/1

# 用规则+噪声生成标签，保证与特征强相关
risk_score = (
    (attendance < 0.7).astype(int) * 2
    + (homework_rate < 0.6).astype(int) * 2
    + (study_hours < 5).astype(int) * 1
    + (past_fail >= 2).astype(int) * 2
    + (online_hours < 3).astype(int) * 1
    + (family_support == 0).astype(int) * 1
)
# 分数>=4 判为高风险，再加少量随机噪声
noise = np.random.choice([0, 1], N, p=[0.9, 0.1])
label = ((risk_score + noise) >= 3).astype(int)

df = pd.DataFrame({
    "student_id": [f"S{i:04d}" for i in range(1, N+1)],
    "attendance": attendance.round(3),
    "homework_rate": homework_rate.round(3),
    "study_hours": study_hours.round(1),
    "past_fail": past_fail,
    "online_hours": online_hours.round(1),
    "family_support": family_support,
    "at_risk": label
})

# 故意制造一点缺失值，方便论文写“缺失值处理”
for col in ["attendance", "homework_rate", "study_hours"]:
    idx = np.random.choice(N, size=int(N*0.02), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("data/students.csv", index=False, encoding="utf-8-sig")
print("已生成 data/students.csv，共", len(df), "条")
print("预警比例：", round(df["at_risk"].mean()*100, 1), "%")
print(df.head())