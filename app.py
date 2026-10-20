import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
st.set_page_config(page_title="智能教务管理系统", layout="wide", page_icon="🎓")

# ================= 简易登录系统 =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎓 智能教务管理系统")
    st.subheader("教师端登录入口")
    with st.form("login_form"):
        username = st.text_input("账号", value="admin")
        password = st.text_input("密码", type="password", value="123456")
        submit = st.form_submit_button("🔐 登录系统")
        if submit:
            if username == "admin" and password == "123456":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("账号或密码错误，默认账号 admin，密码 123456")
    st.stop()

# ================= 侧边栏导航 =================
st.sidebar.title(f"欢迎您，管理员")
st.sidebar.markdown("---")
page = st.sidebar.radio("请选择功能模块", [
    "📊 教务数据看板", 
    "📁 批量数据导入预测",
    "🧑‍🎓 单学生风险预测", 
    "📝 教师干预记录",
    "📈 算法实验评估报告"
])
st.sidebar.markdown("---")
if st.sidebar.button("退出登录"):
    st.session_state.logged_in = False
    st.rerun()
st.sidebar.info("💡 提示：本系统基于随机森林算法与 SHAP 可解释性构建。")

# ================= 数据加载与模型训练 =================
@st.cache_data
def load_data():
    file_path = "data/students.csv"
    try:
        if not os.path.exists(file_path): 
            raise FileNotFoundError("文件不存在")
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        
        # 【关键修复】强制检查列名，如果不对就抛出异常触发自动生成
        required = ["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support", "at_risk"]
        for col in required:
            if col not in df.columns:
                raise ValueError("列名缺失，文件已损坏")
                
    except Exception:
        # 自动生成数据的逻辑
        np.random.seed(42)
        N = 1000
        attendance = np.clip(np.random.normal(0.85, 0.12, N), 0.3, 1.0)
        homework_rate = np.clip(np.random.normal(0.80, 0.15, N), 0.2, 1.0)
        study_hours = np.clip(np.random.normal(8, 3, N), 1, 20)
        past_fail = np.random.poisson(0.8, N)
        online_hours = np.clip(np.random.normal(5, 2, N), 0, 15)
        family_support = np.random.choice([0, 1], N, p=[0.35, 0.65])
        risk_score = ((attendance < 0.8).astype(int) * 2 + (homework_rate < 0.7).astype(int) * 2 + (study_hours < 6).astype(int) * 1 + (past_fail >= 1).astype(int) * 2 + (online_hours < 4).astype(int) * 1 + (family_support == 0).astype(int) * 1)
        noise = np.random.choice([0, 1], N, p=[0.9, 0.1])
        label = ((risk_score + noise) >= 3).astype(int)
        df = pd.DataFrame({"student_id": [f"S{i:04d}" for i in range(1, N+1)], "attendance": attendance.round(3), "homework_rate": homework_rate.round(3), "study_hours": study_hours.round(1), "past_fail": past_fail, "online_hours": online_hours.round(1), "family_support": family_support, "at_risk": label})
    return df.fillna(df.mean(numeric_only=True))

df = load_data()
features = ["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support"]

@st.cache_resource
def train_model(data):
    X_train, _, y_train, _ = train_test_split(data[features], data["at_risk"], test_size=0.2, random_state=42, stratify=data["at_risk"])
    return RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42).fit(X_train, y_train)
model = train_model(df)

def get_risk_level(p):
    if p >= 0.7: return "🔴 高风险"
    elif p >= 0.4: return "🟡 中风险"
    else: return "🟢 低风险"

# ================= 页面 1：整体看板 =================
if page == "📊 教务数据看板":
    st.title("📊 学生学业预警整体看板")
    proba = model.predict_proba(df[features])[:, 1]
    
    df_result = df.copy()
    df_result["风险概率"] = proba.round(3)
    df_result["风险等级"] = df_result["风险概率"].apply(get_risk_level)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("学生总数", len(df_result))
    col2.metric("🔴 高风险", (df_result["风险等级"] == "🔴 高风险").sum())
    col3.metric("🟡 中风险", (df_result["风险等级"] == "🟡 中风险").sum())
    col4.metric("🟢 低风险", (df_result["风险等级"] == "🟢 低风险").sum())
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("风险等级分布")
        counts = df_result["风险等级"].value_counts()
        fig, ax = plt.subplots()
        labels_en = ["High Risk" if "高" in c else ("Medium Risk" if "中" in c else "Low Risk") for c in counts.index]
        ax.pie(counts.values, labels=labels_en, autopct="%1.1f%%", colors=["#e74c3c", "#f39c12", "#2ecc71"][:len(counts)])
        ax.axis("equal")
        st.pyplot(fig)
    with right:
        st.subheader("特征重要性（XAI 归因）")
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        fig2, ax2 = plt.subplots()
        imp.plot(kind="barh", color="steelblue", ax=ax2)
        st.pyplot(fig2)

    st.divider()
    st.subheader("📋 学生名单查询与导出")
    search_id = st.text_input("🔍 输入学号进行实时检索", "")
    view_option = st.radio("数据范围：", ["🔴 仅看高风险", "👥 查看全部"], horizontal=True)
    if view_option == "🔴 仅看高风险":
        show_df = df_result[df_result["风险等级"] == "🔴 高风险"].sort_values("风险概率", ascending=False)
    else:
        show_df = df_result.sort_values("风险概率", ascending=False)
    if search_id:
        show_df = show_df[show_df["student_id"].str.contains(search_id, case=False)]
    st.dataframe(show_df[["student_id", "attendance", "homework_rate", "past_fail", "风险概率", "风险等级"]], use_container_width=True)
    csv = show_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 导出当前名单数据 (CSV)", csv, file_name="students_data.csv", mime="text/csv")

# ================= 页面 2：批量导入预测 =================
elif page == "📁 批量数据导入预测":
    st.title("📁 批量学生数据导入与预测")
    st.caption("支持上传包含学生特征的 CSV 文件，系统将实时调用机器学习模型进行批量预测。")
    
    template_df = pd.DataFrame(columns=features)
    template_df.loc[0] = [0.5, 0.5, 4.0, 2, 3.0, 0]
    template_csv = template_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 下载预测模板 (CSV格式)", template_csv, file_name="template.csv", mime="text/csv")
    
    st.divider()
    uploaded_file = st.file_uploader("请上传 CSV 文件", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            batch_df.columns = batch_df.columns.str.strip()
            
            missing_cols = [col for col in features if col not in batch_df.columns]
            if missing_cols:
                st.error(f"导入失败！文件缺少以下必需的特征列：{missing_cols}")
            else:
                st.success("文件解析成功！开始进行批量预测...")
                probs = model.predict_proba(batch_df[features])[:, 1]
                batch_df["预测风险概率"] = probs.round(3)
                batch_df["预测风险等级"] = batch_df["预测风险概率"].apply(get_risk_level)
                
                st.subheader("📊 批量预测结果")
                st.dataframe(batch_df, use_container_width=True)
                
                result_csv = batch_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("📥 导出带预测结果的数据 (CSV)", result_csv, file_name="prediction_results.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"文件读取失败，请检查文件格式是否正确。错误信息：{e}")

# ================= 页面 3：单学生预测 =================
elif page == "🧑‍🎓 单学生风险预测":
    st.title("🧑‍🎓 单学生风险实时预测")
    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            att = st.number_input("出勤率 (0~1)", 0.0, 1.0, 0.85, 0.01)
            hw = st.number_input("作业提交率 (0~1)", 0.0, 1.0, 0.80, 0.01)
        with col2:
            sh = st.number_input("每周学习时长", 0.0, 50.0, 8.0, 0.5)
            pf = st.number_input("过往挂科数", 0, 20, 0, 1)
        with col3:
            oh = st.number_input("在线学习时长", 0.0, 50.0, 5.0, 0.5)
            fam = st.selectbox("家庭支持", [1, 0], format_func=lambda x: "良好" if x==1 else "较弱")
        submit = st.form_submit_button("🔍 一键评估风险")

    if submit:
        input_data = pd.DataFrame([[att, hw, sh, pf, oh, fam]], columns=features)
        prob = model.predict_proba(input_data)[0, 1]
        level = get_risk_level(prob)
        st.divider()
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("AI 预测风险概率", f"{prob:.1%}")
        col_res2.metric("判定风险等级", level)
        st.subheader("💡 系统自动生成的干预建议")
        suggestions = []
        if att < 0.8: suggestions.append(f"⚠️ 出勤率仅 {att:.1%}，建议辅导员约谈。")
        if hw < 0.7: suggestions.append(f"⚠️ 作业提交率 {hw:.1%}，建议安排学业帮扶。")
        if pf >= 1: suggestions.append(f"⚠️ 已有 {pf} 门挂科记录，建议制定复习计划。")
        if sh < 6: suggestions.append(f"⚠️ 每周学习仅 {sh:.1f} 小时，建议参加晚自习。")
        if fam == 0: suggestions.append("⚠️ 家庭支持较弱，建议加强心理关怀。")
        if not suggestions: st.success("✅ 该生各项指标良好，暂无预警风险！")
        else:
            for s in suggestions: st.warning(s)

# ================= 页面 4：教师干预记录 =================
elif page == "📝 教师干预记录":
    st.title("📝 教师干预记录管理")
    proba = model.predict_proba(df[features])[:, 1]
    high_risk_ids = df[proba >= 0.7]["student_id"].tolist()
    if not high_risk_ids: st.info("当前没有高风险学生，无需干预。")
    else:
        selected_sid = st.selectbox("选择需要干预的学生", high_risk_ids)
        with st.form("intervention_form"):
            action = st.selectbox("干预措施", ["辅导员约谈", "联系家长", "安排学业帮扶", "心理辅导"])
            note = st.text_area("备注详情", "已约定每周五下午辅导。")
            submitted = st.form_submit_button("✅ 提交干预记录")
            if submitted:
                record = pd.DataFrame([[selected_sid, action, note, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]], columns=["学号", "干预措施", "备注", "记录时间"])
                file_path = "data/intervention_records.csv"
                if not os.path.exists("data"): os.makedirs("data")
                if os.path.exists(file_path): record.to_csv(file_path, mode='a', header=False, index=False, encoding="utf-8-sig")
                else: record.to_csv(file_path, mode='w', header=True, index=False, encoding="utf-8-sig")
                st.success(f"已成功记录对 {selected_sid} 的干预措施！")
        st.divider()
        st.subheader("📜 历史干预记录")
        if os.path.exists("data/intervention_records.csv"):
            st.dataframe(pd.read_csv("data/intervention_records.csv"), use_container_width=True)

# ================= 页面 5：算法实验评估报告 =================
elif page == "📈 算法实验评估报告":
    st.title("📈 模型算法对比与可解释性评估")
    tab1, tab2, tab3 = st.tabs(["模型对比与调参", "SMOTE 不平衡处理", "SHAP 可解释性分析"])
    with tab1:
        if os.path.exists("output/roc_curve.png"): st.image("output/roc_curve.png", caption="模型对比")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("output/exp1_tree_depth.png"): st.image("output/exp1_tree_depth.png")
        with col2:
            if os.path.exists("output/exp2_rf_estimators.png"): st.image("output/exp2_rf_estimators.png")
    with tab2:
        if os.path.exists("output/smote_roc.png"): st.image("output/smote_roc.png")
    with tab3:
        if os.path.exists("output/shap_summary.png"): st.image("output/shap_summary.png")
        if os.path.exists("output/shap_waterfall.png"): st.image("output/shap_waterfall.png")
