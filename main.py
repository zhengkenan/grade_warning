import subprocess
import sys

print("====== 开始执行学业预警系统全流程 ======")
print("1. 正在生成/加载脱敏数据...")
subprocess.run([sys.executable, "generate_data.py"])

print("2. 正在计算特征相关性并绘图...")
subprocess.run([sys.executable, "eda.py"])

print("3. 正在执行 SMOTE 数据不平衡处理实验...")
subprocess.run([sys.executable, "smote_experiment.py"])

print("4. 正在计算 SHAP 可解释性归因...")
subprocess.run([sys.executable, "shap_experiment.py"])

print("5. 正在训练并验证最优融合模型...")
subprocess.run([sys.executable, "advanced_experiment.py"])

print("\n====== 全部离线实验与图表生成完毕 ======")
print("请运行 'streamlit run app.py' 启动教师端可视化系统。")