import os
import io
import csv
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =========================================================
# Matplotlib 中文字体配置
# =========================================================
plt.rcParams["axes.unicode_minus"] = False

CHINESE_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

CHINESE_FONT_NAME = None
for font_path in CHINESE_FONT_PATHS:
    if os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
            CHINESE_FONT_NAME = font_manager.FontProperties(
                fname=font_path
            ).get_name()
            break
        except Exception as font_error:
            print(f"中文字体加载失败：{font_path}；原因：{font_error}")

if CHINESE_FONT_NAME:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [CHINESE_FONT_NAME, "DejaVu Sans"]
    print(f"已加载中文字体：{CHINESE_FONT_NAME}")
else:
    print("警告：未找到 Noto CJK 中文字体，请检查 packages.txt 和部署日志。")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# =========================================================
# 1. 页面配置与全局参数
# =========================================================

st.set_page_config(
    page_title="学生学业成绩预警系统",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
RECORD_PATH = os.path.join(BASE_DIR, "data", "intervention_records.csv")

FEATURES = [
    "attendance",
    "homework_rate",
    "study_hours",
    "past_fail",
    "online_hours",
    "family_support"
]

TARGET = "at_risk"
REQUIRED_COLUMNS = ["student_id"] + FEATURES + [TARGET]

FEATURE_LABELS = {
    "attendance": "出勤率",
    "homework_rate": "作业完成率",
    "study_hours": "学习时长",
    "past_fail": "以往挂科情况",
    "online_hours": "在线学习时长",
    "family_support": "家庭支持"
}

RISK_COLORS = {
    "高风险": "#E45756",
    "中风险": "#F2CF5B",
    "低风险": "#54A24B"
}


# =========================================================
# 2. CSV 读取、数据检查与演示数据
# =========================================================

def read_csv_robust(source):
    """读取逗号、制表符或分号分隔的 CSV，并清理列名。"""

    if isinstance(source, str):
        with open(source, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(8192)

        try:
            delimiter = csv.Sniffer().sniff(
                sample, delimiters=",\t;"
            ).delimiter
        except csv.Error:
            delimiter = ","

        df = pd.read_csv(
            source,
            sep=delimiter,
            encoding="utf-8-sig",
            skipinitialspace=True
        )
    else:
        raw = source.getvalue()
        sample = raw[:8192].decode("utf-8-sig", errors="replace")

        try:
            delimiter = csv.Sniffer().sniff(
                sample, delimiters=",\t;"
            ).delimiter
        except csv.Error:
            delimiter = ","

        df = pd.read_csv(
            io.BytesIO(raw),
            sep=delimiter,
            encoding="utf-8-sig",
            skipinitialspace=True
        )

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    return df


def make_demo_data(n=300):
    """仅在仓库不存在学生数据文件时生成演示数据。"""

    rng = np.random.default_rng(42)

    attendance = rng.uniform(0.45, 1.0, n)
    homework_rate = rng.uniform(0.35, 1.0, n)
    study_hours = rng.uniform(0, 6, n)
    past_fail = rng.integers(0, 4, n)
    online_hours = rng.uniform(0, 8, n)
    family_support = rng.integers(0, 2, n)

    score = (
        1.8 * (1 - attendance)
        + 1.5 * (1 - homework_rate)
        + 0.35 * past_fail
        - 0.20 * study_hours
        + 0.06 * online_hours
        - 0.20 * family_support
        + rng.normal(0, 0.25, n)
    )

    threshold = np.median(score)

    return pd.DataFrame({
        "student_id": [f"S{i + 1:04d}" for i in range(n)],
        "attendance": attendance.round(3),
        "homework_rate": homework_rate.round(3),
        "study_hours": study_hours.round(2),
        "past_fail": past_fail,
        "online_hours": online_hours.round(2),
        "family_support": family_support,
        "at_risk": (score >= threshold).astype(int)
    })


@st.cache_data
def load_data():
    """加载并验证学生数据。"""

    if not os.path.exists(DATA_PATH):
        st.warning(
            "未找到 data/students.csv，当前使用自动生成的演示数据。"
        )
        return make_demo_data()

    try:
        df = read_csv_robust(DATA_PATH)
    except Exception as e:
        raise ValueError(f"读取学生数据失败：{e}")

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"学生数据缺少字段：{missing}。"
            f"实际读取到的字段：{df.columns.tolist()}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["student_id"] = df["student_id"].astype(str).str.strip()

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("没有有效的 at_risk 标签，请检查是否为 0 或 1。")

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "at_risk 必须同时包含 0 和 1 两类数据，才能训练分类模型。"
        )

    all_missing = df[FEATURES].isna().all()
    if all_missing.any():
        bad = all_missing[all_missing].index.tolist()
        raise ValueError(f"以下特征列没有有效数值：{bad}")

    if len(df) < 10:
        raise ValueError("有效学生数据少于 10 条，请检查数据文件。")

    return df.reset_index(drop=True)


# =========================================================
# 3. 机器学习模型与特征重要性
# =========================================================

def build_models():
    """建立三种待比较的分类模型。"""

    return {
        "逻辑回归": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "决策树": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ))
        ]),

        "随机森林": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }


@st.cache_resource
def train_model(data):
    """
    在训练集上进行交叉验证和模型比较；
    在独立测试集上评估模型；
    最后使用全部数据训练用于系统预测的模型。
    """

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "风险类别中至少有一类不足 2 条数据，无法进行分层训练。"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    folds = min(5, int(y_train.value_counts().min()))

    if folds < 2:
        raise ValueError("训练数据不足，无法进行交叉验证。")

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    models = build_models()

    rows = []
    roc_data = {}

    best_name = None
    best_cv_auc = -1.0
    best_test_model = None

    for name, estimator in models.items():
        auc_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1
        )

        estimator.fit(X_train, y_train)

        test_prob = estimator.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        test_auc = roc_auc_score(y_test, test_prob)
        fpr, tpr, _ = roc_curve(y_test, test_prob)

        roc_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(test_auc)
        }

        rows.append({
            "模型": name,
            "交叉验证AUC均值": round(float(auc_scores.mean()), 4),
            "测试集准确率": round(
                accuracy_score(y_test, test_pred), 4
            ),
            "测试集精确率": round(
                precision_score(
                    y_test, test_pred, zero_division=0
                ), 4
            ),
            "测试集召回率": round(
                recall_score(
                    y_test, test_pred, zero_division=0
                ), 4
            ),
            "测试集F1": round(
                f1_score(
                    y_test, test_pred, zero_division=0
                ), 4
            ),
            "测试集AUC": round(float(test_auc), 4)
        })

        if float(auc_scores.mean()) > best_cv_auc:
            best_cv_auc = float(auc_scores.mean())
            best_name = name
            best_test_model = estimator

    metrics_df = pd.DataFrame(rows).sort_values(
        "交叉验证AUC均值",
        ascending=False
    ).reset_index(drop=True)

    best_test_prob = best_test_model.predict_proba(X_test)[:, 1]
    best_test_pred = (best_test_prob >= 0.5).astype(int)

    final_model = build_models()[best_name]
    final_model.fit(X, y)

    return (
        final_model,
        best_name,
        metrics_df,
        X_test,
        y_test,
        best_test_prob,
        best_test_pred,
        roc_data
    )


def get_importance(model):
    """提取树模型特征重要性或逻辑回归系数绝对值。"""

    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(FEATURES))

    return pd.DataFrame({
        "特征": FEATURES,
        "特征名称": [FEATURE_LABELS[x] for x in FEATURES],
        "重要性": values
    }).sort_values("重要性", ascending=False)


def risk_level(prob):
    if prob >= 0.7:
        return "高风险"
    if prob >= 0.4:
        return "中风险"
    return "低风险"


def recommendations(row, prob):
    """根据输入特征生成规则性干预建议。"""

    suggestions = []

    if row["attendance"] < 0.8:
        suggestions.append("加强考勤管理，了解缺勤原因。")

    if row["homework_rate"] < 0.7:
        suggestions.append("制定作业完成计划，定期检查学习任务。")

    if row["study_hours"] < 2:
        suggestions.append("帮助学生制定合理的每日学习计划。")

    if row["past_fail"] >= 1:
        suggestions.append("针对薄弱课程安排答疑或补习。")

    if row["family_support"] == 0:
        suggestions.append("加强家校沟通，了解学生的支持需求。")

    if not suggestions:
        suggestions.append("保持现有学习习惯，定期跟踪学习状态。")

    if prob >= 0.7:
        suggestions.insert(
            0, "建议教师优先跟进，并进行个别谈话。"
        )

    return suggestions


# =========================================================
# 4. 登录模块
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎓 学生学业成绩预警系统")
    st.caption("基于机器学习的学生学业风险分析与干预平台")
    st.markdown("---")

    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button(
            "登录",
            use_container_width=True
        )

    if submitted:
        if username == "admin" and password == "123456":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("用户名或密码错误。演示账号：admin，密码：123456")

    st.stop()


# =========================================================
# 5. 导航与系统初始化
# =========================================================

st.sidebar.title("🎓 学业预警系统")
st.sidebar.caption("基于机器学习的风险分析平台")

page = st.sidebar.radio(
    "功能导航",
    [
        "学业预警看板",
        "批量预警",
        "单个学生预警",
        "干预记录",
        "实验报告"
    ]
)

if st.sidebar.button("退出登录"):
    st.session_state.logged_in = False
    st.rerun()

try:
    df = load_data()

    (
        model,
        best_name,
        metrics_df,
        X_test,
        y_test,
        best_test_prob,
        best_test_pred,
        roc_data
    ) = train_model(df)

except Exception as e:
    st.error("系统初始化失败，请检查学生数据和模型训练条件。")
    st.exception(e)
    st.stop()

st.sidebar.success(f"当前最优模型：{best_name}")
st.sidebar.caption(f"当前数据量：{len(df)} 条")


# =========================================================
# 6. 学业预警看板
# =========================================================

if page == "学业预警看板":

    st.title("📊 学业预警看板")
    st.caption("学生风险概览、风险分布与主要影响因素")

    probs = model.predict_proba(df[FEATURES])[:, 1]

    dashboard = df.copy()
    dashboard["风险概率"] = probs
    dashboard["风险等级"] = [
        risk_level(p) for p in probs
    ]

    high_count = int(
        (dashboard["风险等级"] == "高风险").sum()
    )
    medium_count = int(
        (dashboard["风险等级"] == "中风险").sum()
    )
    low_count = int(
        (dashboard["风险等级"] == "低风险").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("学生总数", len(dashboard))
    c2.metric("高风险人数", high_count)
    c3.metric("中风险人数", medium_count)
    c4.metric("低风险人数", low_count)

    st.divider()

    left, right = st.columns([1, 1.25])

    with left:
        st.subheader("风险等级分布")

        risk_counts = pd.Series({
            "高风险": high_count,
            "中风险": medium_count,
            "低风险": low_count
        })

        fig, ax = plt.subplots(figsize=(6, 4.5))

        colors = [
            RISK_COLORS["高风险"],
            RISK_COLORS["中风险"],
            RISK_COLORS["低风险"]
        ]

        if risk_counts.sum() > 0:
            wedges, _, autotexts = ax.pie(
                risk_counts.values,
                labels=None,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                colors=colors,
                pctdistance=0.78,
                wedgeprops={
                    "width": 0.38,
                    "edgecolor": "white"
                }
            )

            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color("#222222")

            ax.legend(
                wedges,
                [
                    f"{name}：{count}人"
                    for name, count in risk_counts.items()
                ],
                title="风险等级",
                loc="lower center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=1,
                frameon=False
            )

            ax.set_title("学生学业风险占比")
            ax.axis("equal")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("各风险等级人数")

        risk_bar = risk_counts.sort_values()

        fig, ax = plt.subplots(figsize=(6, 4.5))

        bars = ax.barh(
            risk_bar.index,
            risk_bar.values,
            color=[
                RISK_COLORS.get(x, "#4C78A8")
                for x in risk_bar.index
            ]
        )

        ax.set_xlabel("学生人数")
        ax.set_title("不同风险等级学生人数")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        for bar in bars:
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {int(bar.get_width())}",
                va="center"
            )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("主要影响因素")

    importance = get_importance(model).sort_values("重要性")

    fig, ax = plt.subplots(figsize=(9, 4.8))

    bars = ax.barh(
        importance["特征名称"],
        importance["重要性"],
        color="#4C78A8"
    )

    ax.set_title("模型特征重要性")
    ax.set_xlabel("相对重要性")
    ax.set_ylabel("学生特征")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bar in bars:
        value = bar.get_width()
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=9
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("学生风险明细")

    search = st.text_input("按学生编号搜索")

    level_filter = st.multiselect(
        "风险等级筛选",
        ["高风险", "中风险", "低风险"],
        default=["高风险", "中风险", "低风险"]
    )

    shown = dashboard[
        dashboard["风险等级"].isin(level_filter)
    ].copy()

    if search.strip():
        shown = shown[
            shown["student_id"].astype(str).str.contains(
                search.strip(),
                case=False,
                na=False
            )
        ]

    shown = shown.sort_values("风险概率", ascending=False)

    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "下载看板数据 CSV",
        dashboard.to_csv(index=False).encode("utf-8-sig"),
        file_name="学业预警看板.csv",
        mime="text/csv"
    )
# =========================================================
# 7. 批量预警
# =========================================================

elif page == "批量预警":

    st.title("📁 批量学生预警")
    st.write("上传包含学生特征的 CSV 文件，系统将批量计算风险概率。")

    st.info(
        "上传文件至少需要包含以下六个特征字段："
        + "、".join(FEATURES)
    )

    template = pd.DataFrame(
        columns=["student_id"] + FEATURES
    )

    st.download_button(
        "下载批量预测模板",
        template.to_csv(index=False).encode("utf-8-sig"),
        file_name="批量预测模板.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader(
        "上传学生数据 CSV",
        type=["csv", "txt"]
    )

    if uploaded is not None:

        try:
            batch = read_csv_robust(uploaded)

            missing = [
                col for col in FEATURES
                if col not in batch.columns
            ]

            if missing:
                st.error(
                    f"上传文件缺少特征字段：{missing}。"
                    f"实际读取到的字段：{batch.columns.tolist()}"
                )

            else:
                for col in FEATURES:
                    batch[col] = pd.to_numeric(
                        batch[col],
                        errors="coerce"
                    )

                invalid = batch[FEATURES].isna().any(axis=1)

                if invalid.any():
                    st.warning(
                        f"有 {int(invalid.sum())} 行存在缺失或非数值特征，"
                        "模型将通过中位数填补缺失值。"
                    )

                probs = model.predict_proba(
                    batch[FEATURES]
                )[:, 1]

                batch["风险概率"] = np.round(probs, 4)
                batch["风险等级"] = [
                    risk_level(p) for p in probs
                ]

                st.subheader("批量预警结果")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "高风险人数",
                    int((batch["风险等级"] == "高风险").sum())
                )

                c2.metric(
                    "中风险人数",
                    int((batch["风险等级"] == "中风险").sum())
                )

                c3.metric(
                    "低风险人数",
                    int((batch["风险等级"] == "低风险").sum())
                )

                st.dataframe(
                    batch,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "下载批量预警结果",
                    batch.to_csv(index=False).encode("utf-8-sig"),
                    file_name="批量预警结果.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"文件处理失败：{e}")


# =========================================================
# 8. 单个学生预警
# =========================================================

elif page == "单个学生预警":

    st.title("👤 单个学生风险预测")
    st.write("输入学生的学习情况，查看模型预测结果及干预建议。")

    with st.form("student_form"):

        student_id = st.text_input(
            "学生编号",
            value="NEW001"
        )

        col1, col2 = st.columns(2)

        with col1:
            attendance = st.slider(
                "出勤率",
                0.0, 1.0, 0.85, 0.01
            )

            homework_rate = st.slider(
                "作业完成率",
                0.0, 1.0, 0.80, 0.01
            )

            study_hours = st.number_input(
                "每日学习时长（小时）",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

        with col2:
            past_fail = st.number_input(
                "以往挂科门数",
                min_value=0,
                max_value=30,
                value=0,
                step=1
            )

            online_hours = st.number_input(
                "每日在线学习时长（小时）",
                min_value=0.0,
                max_value=24.0,
                value=2.0,
                step=0.5
            )

            family_support = st.selectbox(
                "家庭支持情况",
                [1, 0],
                format_func=lambda x: (
                    "支持" if x == 1 else "支持不足"
                )
            )

        submitted = st.form_submit_button(
            "开始预测",
            use_container_width=True
        )

    if submitted:

        one = pd.DataFrame([{
            "attendance": attendance,
            "homework_rate": homework_rate,
            "study_hours": study_hours,
            "past_fail": past_fail,
            "online_hours": online_hours,
            "family_support": family_support
        }])[FEATURES]

        prob = float(
            model.predict_proba(one)[0, 1]
        )

        level = risk_level(prob)

        st.divider()
        st.subheader("预测结果")

        c1, c2 = st.columns(2)

        c1.metric(
            "风险概率",
            f"{prob:.1%}"
        )

        c2.metric(
            "风险等级",
            level
        )

        st.progress(
            float(np.clip(prob, 0, 1))
        )

        st.caption(
            "风险概率来自当前训练模型，不代表学生必然发生学业风险。"
        )

        st.subheader("学生特征分析")

        radar_names = [
            "出勤率",
            "作业完成率",
            "学习时长",
            "历史挂科",
            "在线学习",
            "家庭支持"
        ]

        max_study = max(
            float(df["study_hours"].max()), 1.0
        )

        max_online = max(
            float(df["online_hours"].max()), 1.0
        )

        max_fail = max(
            float(df["past_fail"].max()), 1.0
        )

        radar_values = [
            attendance,
            homework_rate,
            min(study_hours / max_study, 1.0),
            min(past_fail / max_fail, 1.0),
            min(online_hours / max_online, 1.0),
            family_support
        ]

        angles = np.linspace(
            0, 2 * np.pi, len(radar_names),
            endpoint=False
        ).tolist()

        values_closed = radar_values + radar_values[:1]
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(
            figsize=(6, 5),
            subplot_kw={"polar": True}
        )

        ax.plot(
            angles_closed,
            values_closed,
            linewidth=2,
            color="#4C78A8"
        )

        ax.fill(
            angles_closed,
            values_closed,
            color="#4C78A8",
            alpha=0.2
        )

        ax.set_xticks(angles)
        ax.set_xticklabels(radar_names)
        ax.set_ylim(0, 1)
        ax.set_title("学生特征相对水平")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("个性化干预建议")

        for suggestion in recommendations(
            one.iloc[0],
            prob
        ):
            st.write(f"- {suggestion}")

        result = one.copy()
        result.insert(0, "student_id", student_id)
        result["风险概率"] = round(prob, 4)
        result["风险等级"] = level

        st.download_button(
            "下载本次预测结果",
            result.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{student_id}_预警结果.csv",
            mime="text/csv"
        )


# =========================================================
# 9. 干预记录
# =========================================================

elif page == "干预记录":

    st.title("📝 学生干预记录")
    st.caption("记录教师干预措施、处理内容和后续跟进情况。")

    with st.form("record_form", clear_on_submit=True):

        rid = st.text_input("学生编号")
        teacher = st.text_input("负责教师")

        method = st.selectbox(
            "干预方式",
            [
                "个别谈话",
                "课程辅导",
                "家校沟通",
                "学习计划",
                "其他"
            ]
        )

        content = st.text_area("干预内容")
        followup = st.text_area("后续跟进情况")

        saved = st.form_submit_button("保存记录")

    if saved:

        if not rid.strip() or not content.strip():
            st.warning("请填写学生编号和干预内容。")

        else:

            os.makedirs(
                os.path.dirname(RECORD_PATH),
                exist_ok=True
            )

            record = pd.DataFrame([{
                "学生编号": rid.strip(),
                "负责教师": teacher.strip(),
                "干预方式": method,
                "干预内容": content.strip(),
                "后续跟进": followup.strip(),
                "记录时间": pd.Timestamp.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }])

            if os.path.exists(RECORD_PATH):

                try:
                    old = read_csv_robust(RECORD_PATH)

                    record = pd.concat(
                        [old, record],
                        ignore_index=True
                    )

                except Exception as e:
                    st.error(f"原干预记录文件无法读取：{e}")
                    st.stop()

            record.to_csv(
                RECORD_PATH,
                index=False,
                encoding="utf-8-sig"
            )

            st.success("干预记录已保存。")

    st.divider()
    st.subheader("历史干预记录")

    if os.path.exists(RECORD_PATH):

        try:
            records = read_csv_robust(RECORD_PATH)

            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "下载干预记录",
                records.to_csv(index=False).encode("utf-8-sig"),
                file_name="干预记录.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"读取干预记录失败：{e}")

    else:
        st.info("暂时没有干预记录。")

    st.warning(
        "提示：Streamlit Cloud 的运行环境可能重启，"
        "本地文件写入不保证永久保存。正式使用建议连接数据库。"
    )


# =========================================================
# 10. 实验报告与模型分析
# =========================================================

elif page == "实验报告":

    st.title("🧪 模型评估与实验报告")

    st.markdown(
        """
        本模块比较逻辑回归、决策树和随机森林三种分类模型。
        使用分层训练集与测试集划分，并通过训练集交叉验证
        的 AUC 均值选择模型。测试集用于独立性能评估。
        """
    )

    # 一、模型性能指标
    st.subheader("一、模型性能对比")

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    # 二、模型指标分组柱状图
    st.subheader("二、各模型分类指标对比")

    chart_df = metrics_df.set_index("模型")[
        [
            "测试集准确率",
            "测试集精确率",
            "测试集召回率",
            "测试集F1"
        ]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(chart_df.index))
    metric_names = chart_df.columns.tolist()
    width = 0.18

    for i, metric in enumerate(metric_names):
        ax.bar(
            x + (i - (len(metric_names) - 1) / 2) * width,
            chart_df[metric].values,
            width,
            label=metric
        )

    ax.set_xticks(x)
    ax.set_xticklabels(chart_df.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("指标值")
    ax.set_title("不同模型的测试集分类指标")
    ax.legend(ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 三、ROC 曲线
    st.subheader("三、ROC 曲线对比")
    st.caption(
        "ROC 曲线反映模型在不同分类阈值下的区分能力；"
        "曲线越靠近左上角，通常表示区分能力越好。"
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, values in roc_data.items():
        ax.plot(
            values["fpr"],
            values["tpr"],
            linewidth=2,
            label=f"{name}（AUC={values['auc']:.3f}）"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="随机分类参考线"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("假阳性率（FPR）")
    ax.set_ylabel("真正率（TPR）")
    ax.set_title("三种机器学习模型 ROC 曲线")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 四、最优模型混淆矩阵
    st.subheader("四、最优模型混淆矩阵")
    st.caption(
        f"当前根据交叉验证 AUC 选择的模型为：{best_name}。"
        "以下混淆矩阵来自该模型在独立测试集上的预测。"
    )

    cm = confusion_matrix(
        y_test,
        best_test_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        cm,
        cmap="Blues"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "预测无风险",
        "预测有风险"
    ])

    ax.set_yticklabels([
        "实际无风险",
        "实际有风险"
    ])

    ax.set_xlabel("模型预测结果")
    ax.set_ylabel("真实标签")
    ax.set_title("测试集分类结果")

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "#222222"
                ),
                fontsize=14,
                fontweight="bold"
            )

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
        label="学生人数"
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 五、分类报告
    st.subheader("五、最优模型分类评估报告")

    report = classification_report(
        y_test,
        best_test_pred,
        labels=[0, 1],
        target_names=["无风险", "有风险"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    # 六、特征重要性
    st.subheader("六、特征重要性分析")

    importance = get_importance(model).sort_values("重要性")

    fig, ax = plt.subplots(figsize=(9, 4.8))

    bars = ax.barh(
        importance["特征名称"],
        importance["重要性"],
        color="#4C78A8"
    )

    ax.set_xlabel("相对重要性")
    ax.set_title(f"{best_name} 的特征重要性")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bar in bars:
        value = bar.get_width()
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=9
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "说明：树模型的特征重要性与逻辑回归系数绝对值的含义不同，"
        "这里用于辅助理解模型，不应直接解释为因果关系。"
    )

    # 七、导出实验结果
    st.subheader("七、实验数据导出")

    st.download_button(
        "下载模型性能对比 CSV",
        metrics_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="模型性能对比.csv",
        mime="text/csv"
    )

    st.download_button(
        "下载分类评估报告 CSV",
        report_df.to_csv().encode("utf-8-sig"),
        file_name="分类评估报告.csv",
        mime="text/csv"
    )
        # ========================================================
    # 新增：参数实验结果与 SHAP 可解释性分析
    # ========================================================

    st.divider()
    st.header("六、模型参数实验")

    st.markdown(
        """
        本部分通过交叉验证比较不同决策树深度和随机森林树数量的表现。
        主要观察 AUC、召回率与 F1 指标的变化，为模型参数选择提供依据。
        """
    )

    experiment_images = [
        (
            "exp1_tree_depth.png",
            "决策树深度实验",
            "比较不同最大深度下模型的交叉验证 AUC。"
        ),
        (
            "exp1_tree_recall_f1.png",
            "决策树召回率与 F1",
            "观察决策树深度变化对召回率和 F1 的影响。"
        ),
        (
            "exp2_rf_estimators.png",
            "随机森林树数量实验",
            "比较不同决策树数量下随机森林的交叉验证 AUC。"
        ),
        (
            "exp2_rf_recall_f1.png",
            "随机森林召回率与 F1",
            "观察随机森林树数量变化对召回率和 F1 的影响。"
        )
    ]

    for i in range(0, len(experiment_images), 2):
        col1, col2 = st.columns(2)

        for col, item in zip(
            [col1, col2],
            experiment_images[i:i + 2]
        ):
            filename, title, description = item
            image_path = os.path.join(
                BASE_DIR,
                "output",
                filename
            )

            with col:
                st.subheader(title)
                st.caption(description)

                if os.path.exists(image_path):
                    st.image(
                        image_path,
                        use_container_width=True
                    )
                else:
                    st.info(
                        f"尚未生成 {filename}。"
                        "请先运行 experiment.py。"
                    )

    # 展示参数实验的数值结果
    st.subheader("参数实验数值结果")

    experiment_tables = [
        (
            "决策树深度实验数据",
            "exp1_tree_depth_results.csv"
        ),
        (
            "随机森林树数量实验数据",
            "exp2_rf_estimators_results.csv"
        )
    ]

    for title, filename in experiment_tables:
        csv_path = os.path.join(
            BASE_DIR,
            "output",
            filename
        )

        with st.expander(title):
            if os.path.exists(csv_path):
                try:
                    result_df = pd.read_csv(csv_path)
                    st.dataframe(
                        result_df.round(4),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        label=f"下载{title}",
                        data=result_df.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_{filename}"
                    )
                except Exception as e:
                    st.error(f"读取实验结果失败：{e}")
            else:
                st.info(
                    "尚未找到实验结果文件，请先运行 experiment.py。"
                )

    # ========================================================
    # 新增：SHAP 可解释性分析
    # ========================================================

    st.divider()
    st.header("七、SHAP 模型可解释性分析")

    st.markdown(
        """
        SHAP 用于分析不同学生特征对模型输出的影响。

        - **全局影响分析：** 观察各项特征在整体样本中的影响分布。
        - **特征重要性：** 根据平均绝对 SHAP 值比较特征影响程度。
        - **单个学生解释：** 分析某个学生的特征如何推动模型输出上升或下降。

        SHAP 图反映的是模型学到的关联和预测贡献，
        不应直接解释为这些因素与学业风险之间存在因果关系。
        """
    )

    shap_images = [
        (
            "shap_summary.png",
            "SHAP 全局特征影响分析"
        ),
        (
            "shap_feature_importance.png",
            "SHAP 特征重要性"
        ),
        (
            "shap_waterfall.png",
            "单个学生风险解释"
        ),
        (
            "final_feature_importance.png",
            "最终随机森林特征重要性"
        )
    ]

    for i in range(0, len(shap_images), 2):
        col1, col2 = st.columns(2)

        for col, item in zip(
            [col1, col2],
            shap_images[i:i + 2]
        ):
            filename, title = item
            image_path = os.path.join(
                BASE_DIR,
                "output",
                filename
            )

            with col:
                st.subheader(title)

                if os.path.exists(image_path):
                    st.image(
                        image_path,
                        use_container_width=True
                    )
                else:
                    st.info(
                        f"尚未生成 {filename}。"
                        "请先运行 experiment.py。"
                    )

    # 展示最终模型指标与特征重要性 CSV
    st.subheader("最终模型实验数据")

    final_files = [
        (
            "最终随机森林测试集指标",
            "final_model_metrics.csv"
        ),
        (
            "最终模型特征重要性",
            "final_feature_importance.csv"
        ),
        (
            "单个学生 SHAP 贡献明细",
            "shap_single_student_details.csv"
        )
    ]

    for title, filename in final_files:
        csv_path = os.path.join(
            BASE_DIR,
            "output",
            filename
        )

        with st.expander(title):
            if os.path.exists(csv_path):
                try:
                    result_df = pd.read_csv(csv_path)

                    st.dataframe(
                        result_df.round(4),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        label=f"下载{title}",
                        data=result_df.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_final_{filename}"
                    )
                except Exception as e:
                    st.error(f"读取文件失败：{e}")
            else:
                st.info(
                    "暂时没有该文件，请先运行 experiment.py。"
                )

# =========================================================
# 运行 experiment.py
# =========================================================
st.divider()
st.subheader("运行独立实验脚本")
st.write("点击按钮后，执行项目根目录下的 experiment.py。")

if st.button("▶ 运行 experiment.py", key="run_experiment_py"):
    experiment_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment.py"
    )

    if not os.path.isfile(experiment_path):
        st.error("找不到 experiment.py，请确认它已上传到 GitHub 仓库根目录。")
    else:
        with st.spinner("正在运行 experiment.py，请稍候……"):
            try:
                result = subprocess.run(
                    [sys.executable, experiment_path],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if result.returncode == 0:
                    st.success("experiment.py 执行完成！")
                else:
                    st.error(
                        f"experiment.py 执行失败，退出代码：{result.returncode}"
                    )

                if result.stdout.strip():
                    st.text_area(
                        "运行输出",
                        result.stdout[-12000:],
                        height=250
                    )

                if result.stderr.strip():
                    st.text_area(
                        "错误信息",
                        result.stderr[-12000:],
                        height=200
                    )

            except subprocess.TimeoutExpired:
                st.error("运行超过 10 分钟，已停止。")

            except Exception as exc:
                st.error(f"执行 experiment.py 时发生错误：{exc}")

# =========================================================
# 11. 页面底部
# =========================================================

st.sidebar.divider()
st.sidebar.caption("学生学业成绩预警系统")
st.sidebar.caption("毕业设计演示版本")
