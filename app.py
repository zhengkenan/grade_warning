import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
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
st.sidebar.info("💡 提示：本系统基于随机森林算法与 XAI 归因构建。")

# ================= 数据加载与模型训练（史诗级数据内置） =================
@st.cache_data
def load_data():
    # 优先尝试读取本地（如果有的话），如果云端报错，直接触发内置生成
    try:
        file_path = "data/students.csv"
        if not os.path.exists(file_path): raise FileNotFoundError
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df["student_id"] = df["student_id"].astype(str)
        required = ["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support", "at_risk"]
        for col in required:
            if col not in df.columns: raise ValueError("列名缺失")
        return df.fillna(df.mean(numeric_only=True))
    except Exception:
        # 【修改核心：拉大挂科数的权重，让 past_fail 成为绝对的主导因素】
        np.random.seed(42)
        N = 1000
        engagement = np.random.beta(5, 2, N) 
        
        attendance = np.clip(engagement * 0.5 + 0.4 + np.random.normal(0, 0.1, N), 0.2, 1.0)
        homework_rate = np.clip(engagement * 0.6 + 0.3 + np.random.normal(0, 0.1, N), 0.1, 1.0)
        study_hours = np.clip(engagement * 15 + np.random.normal(0, 2, N), 1, 30)
        online_hours = np.clip(engagement * 10 + np.random.normal(0, 1.5, N), 0, 20)
        
        quiz_score = np.clip(engagement * 60 + 30 + np.random.normal(0, 8, N), 0, 100)
        
        # 挂科数分布：如果积极性低，挂科数会显著增加
        past_fail = np.random.poisson(lam=(1 - engagement) * 4.0, size=N) 
        
        family_support = np.random.choice([0, 1], N, p=[0.3, 0.7])
        
        # 【终极改版】挂科一票否决制：只要 past_fail >= 2，直接扣大分
        risk_score = (
            quiz_score * 0.4                     # 成绩占比降低
            + attendance * 10                    # 出勤率
            + homework_rate * 10                 # 作业率
            - past_fail * 15                     # 🔥 挂科惩罚极重：挂1科扣15分，挂2科扣30分
            + family_support * 5                 # 家庭支持
        )
        
        # 加入正态分布随机噪声（模拟突发状况、老师压分等不可控因素）
        noise = np.random.normal(0, 8, N)
        final_score = risk_score + noise
        
        # 如果最终得分低于 55 分，判定为高风险
        at_risk = (final_score < 55).astype(int)
        
        df = pd.DataFrame({
            "student_id": [f"S{i:04d}" for i in range(1, N+1)],
            "attendance": attendance.round(3),
            "homework_rate": homework_rate.round(3),
            "study_hours": study_hours.round(1),
            "past_fail": past_fail,
            "online_hours": online_hours.round(1),
            "family_support": family_support,
            "at_risk": at_risk
        })
        # 模拟少量缺失值
        for col in ["attendance", "homework_rate", "study_hours"]:
            idx = np.random.choice(N, size=int(N*0.015), replace=False)
            df.loc[idx, col] = np.nan
        return df.fillna(df.mean(numeric_only=True))

df = load_data()
features = ["attendance", "homework_rate", "study_hours", "past_fail", "online_hours", "family_support"]

@st.cache_resource
def train_model(data):
    X_train, _, y_train, _ = train_test_split(data[features], data["at_risk"], test_size=0.2, random_state=42, stratify=data["at_risk"])
    # 增加模型复杂度，让模型能更明显地捕捉挂科数的非线性影响
    return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42).fit(X_train, y_train)

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
        # 突出显示挂科数（用红色）
        colors = ['#e74c3c' if x == 'past_fail' else 'steelblue' for x in imp.index]
        imp.plot(kind="barh", color=colors, ax=ax2)
        st.pyplot(fig2)

    st.divider()
    st.subheader("📋 学生名单查询与导出")
    search_id = st.text_input("🔍 输入学号进行实时检索（如 S0001）", "")
    view_option = st.radio("数据范围：", ["🔴 仅看高风险", "👥 查看全部"], horizontal=True)
    if view_option == "🔴 仅看高风险":
        show_df = df_result[df_result["风险等级"] == "🔴 高风险"].sort_values("风险概率", ascending=False)
    else:
        show_df = df_result.sort_values("风险概率", ascending=False)
    if search_id:
        show_df = show_df[show_df["student_id"].str.contains(str(search_id).strip(), case=False)]
    st.dataframe(show_df[["student_id", "attendance", "homework_rate", "past_fail", "风险概率", "风险等级"]], use_container_width=True)
    csv = show_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 导出当前名单数据 (CSV)", csv, file_name="students_data.csv", mime="text/csv")

# ================= 页面 2：批量导入预测 =================
elif page == "📁 批量数据导入预测":
    st.title("📁 批量学生数据导入与预测")
    st.caption("支持上传包含学生特征的 CSV 文件，系统内置脏数据自动清洗与容错机制。")
    
    template_df = pd.DataFrame(columns=features)
    template_df.loc[0] = [0.5, 0.5, 4.0, 2, 3.0, 0]
    st.download_button("📥 下载预测模板 (CSV格式)", template_df.to_csv(index=False, encoding="utf-8-sig"), file_name="template.csv", mime="text/csv")
    
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
                st.success("文件解析成功！正在进行异常值清洗...")
                for col in features:
                    batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce')
                batch_df[features] = batch_df[features].fillna(batch_df[features].median())
                
                batch_df["attendance"] = batch_df["attendance"].clip(0.0, 1.0)
                batch_df["homework_rate"] = batch_df["homework_rate"].clip(0.0, 1.0)
                batch_df["past_fail"] = batch_df["past_fail"].clip(0, 20)
                batch_df["study_hours"] = batch_df["study_hours"].clip(0, 50)
                batch_df["family_support"] = batch_df["family_support"].round().clip(0, 1)
                
                st.info("✅ 数据清洗完成！已自动过滤异常值，开始批量预测...")
                
                probs = model.predict_proba(batch_df[features])[:, 1]
                batch_df["预测风险概率"] = probs.round(3)
                batch_df["预测风险等级"] = batch_df["预测风险概率"].apply(get_risk_level)
                
                st.subheader("📊 批量预测结果")
                st.dataframe(batch_df, use_container_width=True)
                
                result_csv = batch_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("📥 导出带预测结果的数据 (CSV)", result_csv, file_name="prediction_results.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"文件解析严重失败，请检查文件格式是否彻底损坏。错误信息：{e}")

# ================= 页面 3：单学生预测 =================
elif page == "🧑‍🎓 单学生风险预测":
    st.title("🧑‍🎓 单学生风险实时预测与特征画像")
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
        submit = st.form_submit_button("🔍 一键评估风险并生成画像")

    if submit:
        input_data = pd.DataFrame([[att, hw, sh, pf, oh, fam]], columns=features)
        prob = model.predict_proba(input_data)[0, 1]
        level = get_risk_level(prob)
        
        st.divider()
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("AI 预测风险概率", f"{prob:.1%}")
        col_res2.metric("判定风险等级", level)

        st.subheader("🕸️ 学生多维特征画像")
        r_att = att * 100
        r_hw = hw * 100
        r_sh = min(sh / 20 * 100, 100)
        r_pf = max(100 - pf * 20, 0)
        r_oh = min(oh / 15 * 100, 100)
        r_fam = 100 if fam == 1 else 50

        categories = ['出勤率', '作业提交率', '学习时长', '历史成绩(无挂科)', '在线活跃度', '家庭支持']
        values = [r_att, r_hw, r_sh, r_pf, r_oh, r_fam]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.3)',
            line=dict(color='#2ecc71', width=2),
            name='学生特征'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=350
        )
        col_radar1, col_radar2, col_radar3 = st.columns([1, 2, 1])
        with col_radar2:
            st.plotly_chart(fig_radar, use_container_width=True)

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
        else: st.info("云端暂未挂载离线图表，请在本地查看。")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("output/exp1_tree_depth.png"): st.image("output/exp1_tree_depth.png")
        with col2:
            if os.path.exists("output/exp2_rf_estimators.png"): st.image("output/exp2_rf_estimators.png")
    with tab2:
        if os.path.exists("output/smote_roc.png"): st.image("output/smote_roc.png")
    with tab3:
        if os.path.exists("output/shap_summary.png"): st.image("output/shap_summary.png", caption="全局特征影响力蜂群图")
        if os.path.exists("output/shap_waterfall.png"): st.image("output/shap_waterfall.png", caption="单学生风险归因瀑布图")
