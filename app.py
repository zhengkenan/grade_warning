
import os
import io
import csv
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ==================== 1. 基础配置 ====================

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


# ==================== 2. 数据读取与检查 ====================

def read_csv_robust(source):
    """兼容逗号、制表符、分号分隔的数据，并清理列名。"""
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(8192)
        delimiter = csv.Sniffer().sniff(
            sample, delimiters=",\t;"
        ).delimiter if sample.strip() else ","
        df = pd.read_csv(
            source, sep=delimiter, encoding="utf-8-sig",
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
            io.BytesIO(raw), sep=delimiter, encoding="utf-8-sig",
            skipinitialspace=True
        )

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def make_demo_data(n=300):
    """仅当仓库没有学生数据文件时生成演示数据。"""
    rng = np.random.default_rng(42)
    attendance = rng.uniform(0.45, 1.0, n)
    homework = rng.uniform(0.35, 1.0, n)
    study_hours = rng.uniform(0, 6, n)
    past_fail = rng.integers(0, 4, n)
    online_hours = rng.uniform(0, 8, n)
    family_support = rng.integers(0, 2, n)

    score = (
        1.8 * (1 - attendance)
        + 1.5 * (1 - homework)
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
        "homework_rate": homework.round(3),
        "study_hours": study_hours.round(2),
        "past_fail": past_fail,
        "online_hours": online_hours.round(2),
        "family_support": family_support,
        "at_risk": (score >= threshold).astype(int)
    })


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.warning("未找到 data/students.csv，当前使用自动生成的演示数据。")
        return make_demo_data()

    try:
        df = read_csv_robust(DATA_PATH)
    except Exception as e:
        raise ValueError(f"读取学生数据失败：{e}")

    # 再做一次字段名规范化
    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"学生数据缺少字段：{missing}。"
            f"程序实际读取到的字段为：{df.columns.tolist()}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["student_id"] = df["student_id"].astype(str).str.strip()

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("学生数据没有有效的 at_risk 标签，请检查是否为 0 或 1。")

    if df[TARGET].nunique() < 2:
        raise ValueError("at_risk 必须同时包含 0 和 1 两类数据，才能训练分类模型。")

    if df[FEATURES].isna().all().any():
        bad = df[FEATURES].columns[df[FEATURES].isna().all()].tolist()
        raise ValueError(f"以下特征列没有有效数值：{bad}")

    if len(df) < 10:
        raise ValueError("有效学生数据少于 10 条，请检查数据文件。")

    return df.reset_index(drop=True)


# ==================== 3. 模型训练与评估 ====================

def build_models():
    return {
        "逻辑回归": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "决策树": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=5, random_state=42
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
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "风险类别中至少有一类不足 2 条数据，无法进行分层训练。"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    folds = min(5, int(y_train.value_counts().min()))
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=42
    )

    models = build_models()
    rows = []
    best_name = None
    best_auc = -1
    best_model = None

    for name, model in models.items():
        auc_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=1
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        try:
            test_auc = roc_auc_score(y_test, prob)
        except ValueError:
            test_auc = np.nan

        rows.append({
            "模型": name,
            "交叉验证AUC均值": round(float(auc_scores.mean()), 4),
            "测试集准确率": round(accuracy_score(y_test, pred), 4),
            "测试集精确率": round(
                precision_score(y_test, pred, zero_division=0), 4
            ),
            "测试集召回率": round(
                recall_score(y_test, pred, zero_division=0), 4
            ),
            "测试集F1": round(
                f1_score(y_test, pred, zero_division=0), 4
            ),
            "测试集AUC": round(float(test_auc), 4)
            if not np.isnan(test_auc) else np.nan
        })

        if auc_scores.mean() > best_auc:
            best_auc = float(auc_scores.mean())
            best_name = name
            best_model = model

    metrics_df = pd.DataFrame(rows).sort_values(
        "交叉验证AUC均值", ascending=False
    )

    # 使用选中的模型对全部现有数据重新训练，用于系统预测
    best_model.fit(X, y)

    return best_model, best_name, metrics_df, X_test, y_test


def risk_level(prob):
    if prob >= 0.7:
        return "高风险"
    if prob >= 0.4:
        return "中风险"
    return "低风险"


def get_importance(model):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(FEATURES))

    result = pd.DataFrame({
        "影响因素": FEATURES,
        "重要性": values
    })
    return result.sort_values("重要性", ascending=False)


def recommendations(row, prob):
    suggestions = []
    if row["attendance"] < 0.8:
        suggestions.append("加强考勤管理，关注缺勤原因。")
    if row["homework_rate"] < 0.7:
        suggestions.append("制定作业完成计划，定期检查学习任务。")
    if row["study_hours"] < 2:
        suggestions.append("帮助学生制定合理的每日学习计划。")
    if row["past_fail"] >= 1:
        suggestions.append("针对薄弱课程安排答疑或补习。")
    if row["family_support"] == 0:
        suggestions.append("加强家校沟通，了解学生支持需求。")
    if not suggestions:
        suggestions.append("保持现有学习习惯，定期跟踪学习状态。")
    if prob >= 0.7:
        suggestions.insert(0, "建议教师优先跟进，并进行个别谈话。")
    return suggestions


# ==================== 4. 登录 ====================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎓 学生学业成绩预警系统")
    st.caption("基于机器学习的学生学业风险分析与干预平台")

    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True)

    if submitted:
        if username == "admin" and password == "123456":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("用户名或密码错误。演示账号：admin，密码：123456")
    st.stop()


# ==================== 5. 全局加载数据与模型 ====================

st.sidebar.title("🎓 学业预警系统")
st.sidebar.caption("机器学习辅助分析")
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
    model, best_name, metrics_df, X_test, y_test = train_model(df)
except Exception as e:
    st.error("系统初始化失败，请检查学生数据和模型训练条件。")
    st.exception(e)
    st.stop()

st.sidebar.success(f"当前最优模型：{best_name}")


# ==================== 6. 学业预警看板 ====================

if page == "学业预警看板":
    st.title("📊 学业预警看板")
    st.caption("根据学生历史数据和模型预测结果展示风险情况。")

    probs = model.predict_proba(df[FEATURES])[:, 1]
    dashboard = df.copy()
    dashboard["风险概率"] = probs
    dashboard["风险等级"] = [risk_level(p) for p in probs]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("学生总数", len(dashboard))
    c2.metric("高风险人数", int((dashboard["风险等级"] == "高风险").sum()))
    c3.metric("中风险人数", int((dashboard["风险等级"] == "中风险").sum()))
    c4.metric("低风险人数", int((dashboard["风险等级"] == "低风险").sum()))

    st.subheader("风险等级分布")
    risk_counts = dashboard["风险等级"].value_counts().reindex(
        ["高风险", "中风险", "低风险"], fill_value=0
    )
    st.bar_chart(risk_counts)

    st.subheader("学生风险明细")
    search = st.text_input("按学生编号搜索")
    level_filter = st.multiselect(
        "风险等级筛选",
        ["高风险", "中风险", "低风险"],
        default=["高风险", "中风险", "低风险"]
    )
    shown = dashboard[
        dashboard["风险等级"].isin(level_filter)
    ]
    if search.strip():
        shown = shown[
            shown["student_id"].astype(str).str.contains(
                search.strip(), case=False, na=False
            )
        ]

    st.dataframe(
        shown.sort_values("风险概率", ascending=False),
        use_container_width=True
    )

    st.download_button(
        "下载看板数据 CSV",
        dashboard.to_csv(index=False).encode("utf-8-sig"),
        file_name="学业预警看板.csv",
        mime="text/csv"
    )

    st.subheader("主要影响因素")
    importance = get_importance(model)
    st.bar_chart(importance.set_index("影响因素")["重要性"])


# ==================== 7. 批量预警 ====================

elif page == "批量预警":
    st.title("📁 批量学生预警")
    st.write("上传 CSV 文件，文件需包含以下六个特征字段：")
    st.code(",".join(FEATURES))

    template = pd.DataFrame(columns=["student_id"] + FEATURES)
    st.download_button(
        "下载批量预测模板",
        template.to_csv(index=False).encode("utf-8-sig"),
        file_name="批量预测模板.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("上传学生数据 CSV", type=["csv", "txt"])
    if uploaded is not None:
        try:
            batch = read_csv_robust(uploaded)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(
                    f"上传文件缺少特征字段：{missing}。"
                    f"实际读取到：{batch.columns.tolist()}"
                )
            else:
                for col in FEATURES:
                    batch[col] = pd.to_numeric(batch[col], errors="coerce")

                invalid = batch[FEATURES].isna().any(axis=1)
                if invalid.any():
                    st.warning(
                        f"有 {int(invalid.sum())} 行特征缺失或不是数字，"
                        "模型将使用中位数填补缺失值。"
                    )

                probs = model.predict_proba(batch[FEATURES])[:, 1]
                batch["风险概率"] = probs.round(4)
                batch["风险等级"] = [risk_level(p) for p in probs]
                st.dataframe(batch, use_container_width=True)

                st.download_button(
                    "下载批量预警结果",
                    batch.to_csv(index=False).encode("utf-8-sig"),
                    file_name="批量预警结果.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"文件处理失败：{e}")


# ==================== 8. 单个学生预警 ====================

elif page == "单个学生预警":
    st.title("👤 单个学生风险预测")
    st.write("输入学生学习情况，系统将估算其学业风险。")

    with st.form("student_form"):
        student_id = st.text_input("学生编号", value="NEW001")
        col1, col2 = st.columns(2)

        with col1:
            attendance = st.slider("出勤率", 0.0, 1.0, 0.85, 0.01)
            homework_rate = st.slider("作业完成率", 0.0, 1.0, 0.80, 0.01)
            study_hours = st.number_input(
                "每日学习时长（小时）", min_value=0.0,
                max_value=24.0, value=3.0, step=0.5
            )

        with col2:
            past_fail = st.number_input(
                "以往挂科门数", min_value=0, max_value=30,
                value=0, step=1
            )
            online_hours = st.number_input(
                "每日在线学习时长（小时）",
                min_value=0.0, max_value=24.0,
                value=2.0, step=0.5
            )
            family_support = st.selectbox(
                "家庭支持情况", [1, 0],
                format_func=lambda x: "支持" if x == 1 else "支持不足"
            )

        submitted = st.form_submit_button("开始预测", use_container_width=True)

    if submitted:
        one = pd.DataFrame([{
            "attendance": attendance,
            "homework_rate": homework_rate,
            "study_hours": study_hours,
            "past_fail": past_fail,
            "online_hours": online_hours,
            "family_support": family_support
        }])[FEATURES]

        prob = float(model.predict_proba(one)[0, 1])
        level = risk_level(prob)

        st.subheader("预测结果")
        c1, c2 = st.columns(2)
        c1.metric("风险概率", f"{prob:.1%}")
        c2.metric("风险等级", level)

        st.progress(float(np.clip(prob, 0, 1)))
        st.subheader("干预建议")
        row = one.iloc[0]
        for suggestion in recommendations(row, prob):
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


# ==================== 9. 干预记录 ====================

elif page == "干预记录":
    st.title("📝 学生干预记录")
    st.caption("记录教师的跟进措施与后续情况。")

    with st.form("record_form", clear_on_submit=True):
        rid = st.text_input("学生编号")
        teacher = st.text_input("负责教师")
        method = st.selectbox(
            "干预方式",
            ["个别谈话", "课程辅导", "家校沟通", "学习计划", "其他"]
        )
        content = st.text_area("干预内容")
        followup = st.text_area("后续跟进情况")
        saved = st.form_submit_button("保存记录")

    if saved:
        if not rid.strip() or not content.strip():
            st.warning("请填写学生编号和干预内容。")
        else:
            os.makedirs(os.path.dirname(RECORD_PATH), exist_ok=True)
            record = pd.DataFrame([{
                "学生编号": rid.strip(),
                "负责教师": teacher.strip(),
                "干预方式": method,
                "干预内容": content.strip(),
                "后续跟进": followup.strip(),
                "记录时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }])

            if os.path.exists(RECORD_PATH):
                try:
                    old = read_csv_robust(RECORD_PATH)
                    record = pd.concat([old, record], ignore_index=True)
                except Exception:
                    st.error("原干预记录文件无法读取，请检查文件格式。")
                    st.stop()

            record.to_csv(RECORD_PATH, index=False, encoding="utf-8-sig")
            st.success("干预记录已保存。")

    st.subheader("历史干预记录")
    if os.path.exists(RECORD_PATH):
        try:
            records = read_csv_robust(RECORD_PATH)
            st.dataframe(records, use_container_width=True)
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
        "注意：Streamlit Cloud 的运行环境可能重启，"
        "本地写入的记录不保证长期保存。正式使用应连接数据库。"
    )


# ==================== 10. 实验报告 ====================

elif page == "实验报告":
    st.title("🧪 模型评估与实验报告")

    st.subheader("实验说明")
    st.write(
        "系统比较逻辑回归、决策树和随机森林三种分类模型。"
        "数据采用分层训练集与测试集划分，并在训练集上进行交叉验证。"
        "交叉验证 AUC 均值用于选择系统当前使用的模型。"
    )

    st.subheader("模型性能对比")
    st.dataframe(metrics_df, use_container_width=True)

    chart_df = metrics_df.set_index("模型")[
        ["测试集准确率", "测试集精确率", "测试集召回率", "测试集F1"]
    ]
    st.bar_chart(chart_df)

    st.subheader("当前模型信息")
    st.write(f"交叉验证 AUC 最优模型：**{best_name}**")

    st.subheader("测试集混淆矩阵")
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["实际无风险", "实际有风险"],
        columns=["预测无风险", "预测有风险"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("分类评估报告")
    report = classification_report(
        y_test, test_pred,
        labels=[0, 1],
        target_names=["无风险", "有风险"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    st.subheader("特征重要性")
    importance = get_importance(model)
    st.bar_chart(importance.set_index("影响因素")["重要性"])

    st.download_button(
        "下载模型对比结果",
        metrics_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="模型评估结果.csv",
        mime="text/csv"
    )

st.sidebar.caption("学生学业成绩预警系统 · 毕业设计演示版")
