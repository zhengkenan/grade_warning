🎓 基于机器学习的学生学业成绩预警系统

📖 项目简介
本项目是一个端到端的学业预警解决方案，包含从数据模拟、特征工程、多模型对比调参、SMOTE不平衡处理、SHAP可解释性分析，到最终 Web 可视化大屏与业务干预闭环的完整流程。

🛠️ 技术栈
- 核心算法：逻辑回归、决策树、随机森林、Voting 融合、SMOTE、SHAP
- 开发框架：Python 3.12, Pandas, Scikit-learn, Imbalanced-learn
- 前端展示：Streamlit
- 部署环境：Streamlit Community Cloud

📂 目录结构


🚀 运行指南
1. 安装依赖：`pip install -r requirements.txt`
2. 一键跑通离线实验（生成模型与所有图表）：`python main.py`
3. 启动可视化前端系统：`streamlit run app.py`

🌐 在线体验
系统已部署至云端：https://gradewarning-zhengkaixin.streamlit.app/
