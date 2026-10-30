
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ================= 1. 页面基础设置 =================
RANDOM_STATE = 42

FEATURES = [
    "attendance",
    "homework_rate",
    "study_hours",
    "past_fail",
    "online_hours",
    "family_support"
]

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="智能教务管理系统",
    layout="wide",
    page_icon="🎓"
)

# ================= 2. 登录模块 =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎓 智能教务管理系统")
    st.subheader("教师端登录入口")

    with st.form("login_form"):
        username = st.text_input("账号", value="admin")
        password = st.text_input(
            "密码",
            type="password",
            value="123456"
        )
        submitted = st.form_submit_button("🔐 登录系统")

        if submitted:
            if username == "admin" and password == "123456":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("账号或密码错误。")

    st.stop()

# ================= 3. 侧边栏导航 =================
st.sidebar.title("欢迎您，管理员")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "请选择功能模块",
    [
        "📊 教务数据看板",
        "📁 批量数据导入预测",
        "🧑‍🎓 单学生风险预测",
        "📝 教师干预记录",
        "📈 算法实验评估报告"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("退出登录"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.info(
    "系统使用机器学习模型进行学生学业风险预测。"
)

# ================= 4. 数据加载模块 =================
@st.cache_data
def load_data():
    file_path = "data/students.csv"

    if os.path.exists(file_path):
        data = pd.read_csv(
            file_path,
            encoding="utf-8-sig"
        )
        data.columns = data.columns.str.strip()

        required = ["student_id"] + FEATURES + ["at_risk"]
        missing = [
            col for col in required
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"学生数据缺少必要字段：{missing}"
            )
    else:
        # 无数据文件时生成演示数据。
        # 正式实验应优先使用仓库中的 data/students.csv。
        rng = np.random.default_rng(RANDOM_STATE)
        n = 1000

        engagement = rng.beta(5, 2, n)

        attendance = np.clip(
            engagement * 0.5 + 0.4
            + rng.normal(0, 0.1, n),
            0.2, 1.0
        )
        homework_rate = np.clip(
            engagement * 0.6 + 0.3
            + rng.normal(0, 0.1, n),
            0.1, 1.0
        )
        study_hours = np.clip(
            engagement * 15 + rng.normal(0, 2, n),
            1, 30
        )
        online_hours = np.clip(
            engagement * 10 + rng.normal(0, 1.5, n),
            0, 20
        )
        past_fail = rng.poisson(
            (1 - engagement) * 3.0
        )
        family_support = rng.choice(
            [0, 1], n, p=[0.3, 0.7]
        )

        risk_score = (
            (attendance < 0.6).astype(int) * 20
            + (attendance < 0.8).astype(int) * 10
            + (homework_rate < 0.6).astype(int) * 20
            + (homework_rate < 0.8).astype(int) * 10
            + (study_hours < 5).astype(int) * 10
            + (online_hours < 3).astype(int) * 5
            + (family_support == 0).astype(int) * 5
            + past_fail * 30
        )

        risk_score = risk_score + rng.normal(0, 10, n)
        at_risk = (risk_score >= 40).astype(int)

        data = pd.DataFrame({
            "student_id": [
                f"S{i:04d}" for i in range(1, n + 1)
            ],
            "attendance": attendance.round(3),
            "homework_rate": homework_rate.round(3),
            "study_hours": study_hours.round(1),
            "past_fail": past_fail,
            "online_hours": online_hours.round(1),
            "family_support": family_support,
            "at_risk": at_risk
        })

    data["student_id"] = data["student_id"].astype(str)

    for col in FEATURES + ["at_risk"]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna(subset=["at_risk"]).copy()
    data["at_risk"] = data["at_risk"].astype(int)

    if not set(data["at_risk"].unique()).issubset({0, 1}):
        raise ValueError("at_risk 标签必须为 0 或 1。")

    if data["at_risk"].nunique() != 2:
        raise ValueError("数据必须同时包含风险和非风险两类样本。")

    if data[FEATURES].isna().all(axis=0).any():
        raise ValueError("某些特征列全部缺失，请检查数据。")

    return data


df = load_data()
features = FEATURES

# ================= 5. 模型训练与选择模块 =================
def build_models():
    return {
        "逻辑回归": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),
        "决策树": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                random_state=RANDOM_STATE
            ))
        ]),
        "随机森林": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }


@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data["at_risk"].astype(int).copy()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    class_counts = y_train.value_counts()
    n_splits = min(5, int(class_counts.min()))

    if n_splits < 2:
        raise ValueError(
            "训练集某一类别样本过少，无法进行分层交叉验证。"
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = build_models()
    results = []
    best_name = None
    best_auc = -1.0
    best_model = None

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "auc": "roc_auc"
            },
            n_jobs=-1,
            error_score="raise"
        )

        mean_auc = float(scores["test_auc"].mean())

        results.append({
            "模型": name,
            "交叉验证准确率": scores["test_accuracy"].mean(),
            "交叉验证精确率": scores["test_precision"].mean(),
            "交叉验证召回率": scores["test_recall"].mean(),
            "交叉验证F1": scores["test_f1"].mean(),
            "交叉验证AUC": mean_auc
        })

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_name = name
            best_model = clone(pipeline)

    best_model.fit(X_train, y_train)

    results_df = pd.DataFrame(results).round(4)

    return best_model, best_name, best_auc, results_df


model, selected_model, cv_auc, cv_results_df = train_model(df)


def get_risk_level(probability):
    if probability >= 0.7:
        return "🔴 高风险"
    elif probability >= 0.4:
        return "🟡 中风险"
    return "🟢 低风险"


def get_feature_importance(trained_model):
    """仅对具有 feature_importances_ 的树模型展示重要性。"""
    final_model = trained_model.named_steps["model"]

    if hasattr(final_model, "feature_importances_"):
        return pd.Series(
            final_model.feature_importances_,
            index=features
        ).sort_values()

    if hasattr(final_model, "coef_"):
        return pd.Series(
            np.abs(final_model.coef_[0]),
            index=features
        ).sort_values()

    return None


# ================= 页面 1：教务数据看板 =================
if page == "📊 教务数据看板":
    st.title("📊 学生学业预警整体看板")

    st.caption(
        f"当前候选模型：{selected_model}；"
        f"训练集交叉验证 AUC：{cv_auc:.3f}"
    )

    proba = model.predict_proba(df[features])[:, 1]

    df_result = df.copy()
    df_result["风险概率"] = proba.round(3)
    df_result["风险等级"] = (
        df_result["风险概率"].apply(get_risk_level)
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("学生总数", len(df_result))
    col2.metric(
        "🔴 高风险",
        (df_result["风险等级"] == "🔴 高风险").sum()
    )
    col3.metric(
        "🟡 中风险",
        (df_result["风险等级"] == "🟡 中风险").sum()
    )
    col4.metric(
        "🟢 低风险",
        (df_result["风险等级"] == "🟢 低风险").sum()
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("风险等级分布")
        order = ["🔴 高风险", "🟡 中风险", "🟢 低风险"]
        counts = (
            df_result["风险等级"]
            .value_counts()
            .reindex(order, fill_value=0)
        )

        fig, ax = plt.subplots()
        ax.pie(
            counts.values,
            labels=["高风险", "中风险", "低风险"],
            autopct="%1.1f%%",
            colors=["#e74c3c", "#f39c12", "#2ecc71"]
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("特征重要性 / 模型系数")
        importance = get_feature_importance(model)

        if importance is not None:
            fig2, ax2 = plt.subplots()
            importance.plot(kind="barh", ax=ax2)
            ax2.set_xlabel("相对重要性 / 系数绝对值")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("当前模型暂不支持该特征重要性展示。")

    st.divider()
    st.subheader("📋 学生名单查询与导出")

    search_id = st.text_input(
        "输入学号检索，例如 S0001",
        ""
    )
    view_option = st.radio(
        "数据范围",
        ["🔴 仅看高风险", "👥 查看全部"],
        horizontal=True
    )

    if view_option == "🔴 仅看高风险":
        show_df = df_result[
            df_result["风险等级"] == "🔴 高风险"
        ].copy()
    else:
        show_df = df_result.copy()

    if search_id.strip():
        show_df = show_df[
            show_df["student_id"].str.contains(
                search_id.strip(),
                case=False,
                na=False
            )
        ]

    show_df = show_df.sort_values(
        "风险概率", ascending=False
    )

    display_cols = [
        "student_id",
        "attendance",
        "homework_rate",
        "past_fail",
        "风险概率",
        "风险等级"
    ]
    st.dataframe(
        show_df[display_cols],
        use_container_width=True
    )

    st.download_button(
        "📥 导出当前名单数据 CSV",
        show_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="students_data.csv",
        mime="text/csv"
    )


# ================= 页面 2：批量数据导入预测 =================
elif page == "📁 批量数据导入预测":
    st.title("📁 批量学生数据导入与预测")
    st.caption(
        "上传包含六项学生特征的 CSV 文件，"
        "系统将进行数据检查并输出风险预测结果。"
    )

    template_df = pd.DataFrame({
        "attendance": [0.5],
        "homework_rate": [0.5],
        "study_hours": [4.0],
        "past_fail": [2],
        "online_hours": [3.0],
        "family_support": [0]
    })

    st.download_button(
        "📥 下载预测模板",
        template_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="prediction_template.csv",
        mime="text/csv"
    )

    st.divider()
    uploaded_file = st.file_uploader(
        "请上传 CSV 文件",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(
                uploaded_file,
                encoding="utf-8-sig"
            )
            batch_df.columns = batch_df.columns.str.strip()

            missing_cols = [
                col for col in features
                if col not in batch_df.columns
            ]

            if missing_cols:
                st.error(
                    f"文件缺少必需的特征列：{missing_cols}"
                )
            elif batch_df.empty:
                st.warning("上传文件没有数据。")
            else:
                for col in features:
                    batch_df[col] = pd.to_numeric(
                        batch_df[col],
                        errors="coerce"
                    )

                if batch_df[features].isna().all(axis=0).any():
                    st.error("部分特征列全部为空，请检查文件。")
                else:
                    batch_df["attendance"] = (
                        batch_df["attendance"].clip(0, 1)
                    )
                    batch_df["homework_rate"] = (
                        batch_df["homework_rate"].clip(0, 1)
                    )
                    batch_df["past_fail"] = (
                        batch_df["past_fail"].clip(0, 20)
                    )
                    batch_df["study_hours"] = (
                        batch_df["study_hours"].clip(0, 50)
                    )
                    batch_df["online_hours"] = (
                        batch_df["online_hours"].clip(0, 50)
                    )
                    batch_df["family_support"] = (
                        batch_df["family_support"].round().clip(0, 1)
                    )

                    probs = model.predict_proba(
                        batch_df[features]
                    )[:, 1]

                    batch_df["预测风险概率"] = probs.round(3)
                    batch_df["预测风险等级"] = (
                        batch_df["预测风险概率"].apply(get_risk_level)
                    )

                    st.success(
                        f"预测完成，共处理 {len(batch_df)} 条记录。"
                    )
                    st.dataframe(
                        batch_df,
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 导出预测结果 CSV",
                        batch_df.to_csv(
                            index=False,
                            encoding="utf-8-sig"
                        ),
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"文件处理失败：{e}")


# ================= 页面 3：单学生风险预测 =================
elif page == "🧑‍🎓 单学生风险预测":
    st.title("🧑‍🎓 单学生风险预测与特征画像")

    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            att = st.number_input(
                "出勤率（0～1）",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.01
            )
            hw = st.number_input(
                "作业提交率（0～1）",
                min_value=0.0,
                max_value=1.0,
                value=0.80,
                step=0.01
            )

        with col2:
            sh = st.number_input(
                "每周学习时长",
                min_value=0.0,
                max_value=50.0,
                value=8.0,
                step=0.5
            )
            pf = st.number_input(
                "过往挂科数",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

        with col3:
            oh = st.number_input(
                "在线学习时长",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5
            )
            fam = st.selectbox(
                "家庭支持",
                [1, 0],
                format_func=lambda x: "良好" if x == 1 else "较弱"
            )

        submit = st.form_submit_button(
            "🔍 评估风险并生成画像"
        )

    if submit:
        input_data = pd.DataFrame(
            [[att, hw, sh, pf, oh, fam]],
            columns=features
        )

        prob = model.predict_proba(input_data)[0, 1]
        level = get_risk_level(prob)

        st.divider()
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("预测风险概率", f"{prob:.1%}")
        col_res2.metric("风险等级", level)

        st.subheader("🕸️ 学生多维特征画像")

        values = [
            att * 100,
            hw * 100,
            min(sh / 20 * 100, 100),
            max(100 - pf * 20, 0),
            min(oh / 15 * 100, 100),
            100 if fam == 1 else 50
        ]

        categories = [
            "出勤率",
            "作业提交率",
            "学习时长",
            "历史成绩",
            "在线学习",
            "家庭支持"
        ]

        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="学生特征"
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=400
        )

        st.plotly_chart(
            fig_radar,
            use_container_width=True
        )

        st.subheader("💡 系统干预建议")
        suggestions = []

        if att < 0.8:
            suggestions.append(
                f"出勤率为 {att:.1%}，建议教师了解缺勤原因。"
            )
        if hw < 0.7:
            suggestions.append(
                f"作业提交率为 {hw:.1%}，建议安排学业帮扶。"
            )
        if pf >= 1:
            suggestions.append(
                f"已有 {pf} 门挂科记录，建议制定复习计划。"
            )
        if sh < 6:
            suggestions.append(
                f"每周学习时长为 {sh:.1f} 小时，建议加强学习规划。"
            )
        if fam == 0:
            suggestions.append(
                "家庭支持指标较弱，建议教师加强沟通与关怀。"
            )

        if not suggestions:
            st.success("各项输入指标较好，可继续保持并定期跟踪。")
        else:
            for suggestion in suggestions:
                st.warning(suggestion)


# ================= 页面 4：教师干预记录 =================
elif page == "📝 教师干预记录":
    st.title("📝 教师干预记录管理")

    probabilities = model.predict_proba(df[features])[:, 1]
    high_risk_ids = df.loc[
        probabilities >= 0.7,
        "student_id"
    ].astype(str).tolist()

    if not high_risk_ids:
        st.info("当前没有达到高风险阈值的学生。")
    else:
        selected_sid = st.selectbox(
            "选择需要干预的学生",
            high_risk_ids
        )

        with st.form("intervention_form"):
            action = st.selectbox(
                "干预措施",
                [
                    "辅导员约谈",
                    "联系家长",
                    "安排学业帮扶",
                    "心理辅导"
                ]
            )
            note = st.text_area("备注详情")
            submitted = st.form_submit_button("✅ 提交干预记录")

            if submitted:
                record = pd.DataFrame(
                    [[
                        selected_sid,
                        action,
                        note,
                        pd.Timestamp.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ]],
                    columns=["学号", "干预措施", "备注", "记录时间"]
                )

                os.makedirs("data", exist_ok=True)
                file_path = "data/intervention_records.csv"

                if os.path.exists(file_path):
                    record.to_csv(
                        file_path,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf-8-sig"
                    )
                else:
                    record.to_csv(
                        file_path,
                        index=False,
                        encoding="utf-8-sig"
                    )

                st.success(
                    f"已记录对学生 {selected_sid} 的干预措施。"
                )

    st.divider()
    st.subheader("📜 历史干预记录")

    records_path = "data/intervention_records.csv"

    if os.path.exists(records_path):
        try:
            records = pd.read_csv(records_path)
            st.dataframe(records, use_container_width=True)
        except Exception as e:
            st.warning(f"无法读取干预记录：{e}")
    else:
        st.info(
            "暂无干预记录。注意：云端应用重启后，"
            "本地文件记录可能不会保留。"
        )


# ================= 页面 5：算法实验评估报告 =================
elif page == "📈 算法实验评估报告":
    st.title("📈 模型算法对比与实验评估")

    st.caption(
        f"当前网页选择的候选模型：{selected_model}；"
        f"训练集交叉验证 AUC：{cv_auc:.3f}"
    )

    tab1, tab2, tab3 = st.tabs([
        "模型对比",
        "不平衡处理实验",
        "可解释性分析"
    ])

    with tab1:
        st.subheader("三种模型的交叉验证结果")
        st.dataframe(
            cv_results_df,
            use_container_width=True
        )

        st.subheader("交叉验证 AUC 对比")
        chart_df = cv_results_df.set_index("模型")[
            ["交叉验证AUC"]
        ]
        st.bar_chart(chart_df)

        roc_path = "output/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, caption="独立测试集 ROC 曲线对比")
        else:
            st.info(
                "暂未找到 output/roc_curve.png。"
                "请运行 train.py 生成实验图表后再查看。"
            )

        cm_path = "output/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="最佳候选模型混淆矩阵")

        metrics_path = "output/model_metrics.csv"
        if os.path.exists(metrics_path):
            st.subheader("独立测试集评估指标")
            st.dataframe(
                pd.read_csv(metrics_path),
                use_container_width=True
            )

        for filename, caption in [
            ("exp1_tree_depth.png", "决策树深度实验"),
            ("exp2_rf_estimators.png", "随机森林树数量实验")
        ]:
            path = os.path.join("output", filename)
            if os.path.exists(path):
                st.image(path, caption=caption)

    with tab2:
        smote_path = "output/smote_roc.png"
        if os.path.exists(smote_path):
            st.image(smote_path, caption="SMOTE 不平衡处理实验")
        else:
            st.info(
                "暂未找到 SMOTE 实验图表。"
                "运行对应实验脚本后可在此查看。"
            )

    with tab3:
        importance_path = "output/feature_importance.png"
        if os.path.exists(importance_path):
            st.image(
                importance_path,
                caption="随机森林特征重要性"
            )
        else:
            importance = get_feature_importance(model)
            if importance is not None:
                st.subheader("当前模型特征重要性或系数")
                st.bar_chart(importance)

        for filename, caption in [
            ("shap_summary.png", "SHAP 全局特征影响图"),
            ("shap_waterfall.png", "SHAP 单样本风险归因图")
        ]:
            path = os.path.join("output", filename)
            if os.path.exists(path):
                st.image(path, caption=caption)
            else:
                st.info(f"尚未生成：{caption}")
