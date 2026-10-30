
# -*- coding: utf-8 -*-
"""
学生学业成绩预警系统
功能：
1. 学业预警看板
2. 批量学生预警
3. 单个学生预警
4. 干预记录管理
5. 实验报告查看

数据文件：data/students.csv
模型文件：models/final_model.pkl
模型配置：models/config.json
"""

import os
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 一、页面配置
# ============================================================

st.set_page_config(
    page_title="学生学业成绩预警系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")
INTERVENTION_PATH = os.path.join(
    BASE_DIR, "data", "intervention_records.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

FEATURES = [
    "attendance",
    "homework_rate",
    "study_hours",
    "past_fail",
    "online_hours",
    "family_support",
]

FEATURE_NAMES = {
    "attendance": "出勤率",
    "homework_rate": "作业完成率",
    "study_hours": "学习时长",
    "past_fail": "以往挂科情况",
    "online_hours": "线上学习时长",
    "family_support": "家庭支持程度",
}

TARGET = "at_risk"

os.makedirs(os.path.dirname(INTERVENTION_PATH), exist_ok=True)

# 中文图表字体
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 二、通用函数
# ============================================================

def show_header():
    st.title("🎓 学生学业成绩预警系统")
    st.caption(
        "基于机器学习的学业风险识别、风险分析与学生干预管理"
    )


def load_config():
    """读取训练阶段保存的模型配置。"""
    if not os.path.exists(CONFIG_PATH):
        return {}

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config = json.load(file)
        return config if isinstance(config, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_student_data():
    """读取并校验学生数据。"""
    if not os.path.exists(DATA_PATH):
        st.error(f"找不到学生数据文件：{DATA_PATH}")
        st.info(
            "请确认仓库中存在 data/students.csv，"
            "且文件包含六个特征列和 at_risk 标签列。"
        )
        st.stop()

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        st.error(f"读取学生数据失败：{exc}")
        st.stop()

    # 兼容部分文件使用制表符分隔的情况
    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        try:
            df = pd.read_csv(DATA_PATH, sep="\t")
        except Exception:
            pass

    # 去除列名两端的空格
    df.columns = df.columns.astype(str).str.strip()

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        st.error(f"学生数据缺少必要字段：{', '.join(missing)}")
        st.write("当前读取到的字段：", list(df.columns))
        st.stop()

    # 特征和标签转成数值类型
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 标签缺失的记录不能用于监督学习
    df = df.dropna(subset=[TARGET]).copy()

    # 只保留二分类标签 0 和 1
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    if df.empty:
        st.error("清洗后没有可用的学生数据。")
        st.stop()

    # 清理无穷值；特征缺失值在模型 Pipeline 中填补
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    if df[TARGET].nunique() < 2:
        st.error(
            "at_risk 标签只有一个类别，无法训练二分类模型。"
        )
        st.stop()

    if df[FEATURES].isna().all(axis=1).all():
        st.error("所有学生的特征均为空，无法进行预测。")
        st.stop()

    # 如果原始数据没有学号，生成一个展示用编号
    if "student_id" not in df.columns:
        df.insert(
            0,
            "student_id",
            [f"STU{i + 1:04d}" for i in range(len(df))]
        )
    else:
        df["student_id"] = df["student_id"].fillna("").astype(str)
        empty_id = df["student_id"].str.strip().eq("")
        df.loc[empty_id, "student_id"] = [
            f"STU{i + 1:04d}" for i in range(empty_id.sum())
        ]

    return df.reset_index(drop=True)


def get_risk_thresholds(config):
    """
    优先读取训练配置中的阈值。

    支持配置：
    high_threshold / medium_threshold
    decision_threshold / threshold

    若没有配置，使用明确标注的演示默认值。
    """
    high = config.get(
        "high_threshold",
        config.get("decision_threshold", config.get("threshold", 0.5)),
    )
    medium = config.get("medium_threshold", 0.3)

    try:
        high = float(high)
        medium = float(medium)
    except (TypeError, ValueError):
        high, medium = 0.5, 0.3

    high = float(np.clip(high, 0.01, 0.99))
    medium = float(np.clip(medium, 0.0, 0.98))

    if medium >= high:
        medium = max(0.0, high - 0.1)

    return high, medium


def get_positive_probability(model, X):
    """获取 at_risk=1 对应的预测概率。"""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "当前模型没有 predict_proba 方法，"
            "请检查训练脚本导出的模型。"
        )

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("模型类别中没有 at_risk=1。")

    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def classify_risk(probability, high_threshold, medium_threshold):
    """将风险概率划分为高、中、低三个等级。"""
    if probability >= high_threshold:
        return "高风险"
    if probability >= medium_threshold:
        return "中风险"
    return "低风险"


def risk_advice(row, risk_level):
    """根据风险等级和学生特征生成辅助建议。"""
    advice = []

    if risk_level == "高风险":
        advice.append("建议辅导员优先联系学生，核实近期学习情况。")
    elif risk_level == "中风险":
        advice.append("建议持续跟踪学习表现，安排阶段性检查。")
    else:
        advice.append("建议维持当前学习支持，并定期复查。")

    attendance = row.get("attendance", np.nan)
    homework = row.get("homework_rate", np.nan)
    study_hours = row.get("study_hours", np.nan)
    past_fail = row.get("past_fail", np.nan)

    if pd.notna(attendance) and attendance < 0.8:
        advice.append("出勤指标偏低，建议核查缺勤原因。")

    if pd.notna(homework) and homework < 0.6:
        advice.append("作业完成情况偏低，建议制定补交计划。")

    if pd.notna(study_hours) and study_hours < 2:
        advice.append("学习时长较少，可协助制定每周学习计划。")

    if pd.notna(past_fail) and past_fail > 0:
        advice.append("存在以往挂科记录，建议关注相关课程的补修情况。")

    return "；".join(advice)


# ============================================================
# 三、模型加载和备用训练
# ============================================================

@st.cache_resource(show_spinner="正在加载学业预警模型……")
def load_model_and_config(data_path, model_path, config_path):
    """
    优先加载正式训练模型。
    如果模型文件不存在，则使用备用网格搜索训练。

    正式实验建议先运行独立训练脚本，生成 final_model.pkl
    和 config.json，再部署本应用。
    """
    config = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
        except Exception:
            config = {}

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)

            if isinstance(model, dict) and "model" in model:
                # 兼容以字典形式保存模型的情况
                model = model["model"]

            if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                return model, config, "已加载训练阶段保存的模型"

        except Exception as exc:
            st.warning(
                f"正式模型加载失败，将尝试备用训练。原因：{exc}"
            )

    # 备用训练：只在模型文件不存在或无法加载时运行
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件：{data_path}")

    df = pd.read_csv(data_path)

    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        df = pd.read_csv(data_path, sep="\t")

    df.columns = df.columns.astype(str).str.strip()

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"备用训练缺少字段：{missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("每个风险类别至少需要两条记录才能进行备用训练。")

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3],
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X, y)

    # 使用交叉验证召回率选出的最佳参数，在全部可用数据上重训
    model = search.best_estimator_

    config = {
        **config,
        "high_threshold": config.get(
            "high_threshold",
            config.get("decision_threshold", 0.5),
        ),
        "medium_threshold": config.get("medium_threshold", 0.3),
        "model_source": "备用网格搜索模型",
        "best_params": search.best_params_,
        "cv_recall": float(search.best_score_),
    }

    return model, config, "备用模型：随机森林网格搜索（以召回率为选择指标）"


@st.cache_data(show_spinner=False)
def get_model_predictions(data_signature, _model, df):
    """为当前数据生成模型预测结果。"""
    X = df[FEATURES]
    probabilities = get_positive_probability(_model, X)
    return probabilities


# ============================================================
# 四、数据准备
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_data(df, probabilities, high_threshold, medium_threshold):
    result = df.copy()

    result["risk_probability"] = probabilities
    result["risk_level"] = result["risk_probability"].apply(
        lambda p: classify_risk(
            p, high_threshold, medium_threshold
        )
    )

    result["risk_probability_pct"] = (
        result["risk_probability"] * 100
    ).round(2)

    result["risk_advice"] = result.apply(
        lambda row: risk_advice(row, row["risk_level"]),
        axis=1,
    )

    return result


def format_feature_table(df):
    """将内部特征名转成中文展示名。"""
    table = df.copy()
    return table.rename(columns=FEATURE_NAMES)


def show_risk_distribution(df, column="risk_level"):
    counts = (
        df[column]
        .value_counts()
        .reindex(["高风险", "中风险", "低风险"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#D9534F", "#E6A23C", "#49A078"],
    )

    ax.set_title("学生风险等级分布")
    ax.set_xlabel("风险等级")
    ax.set_ylabel("学生人数")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_feature_importance(model):
    """展示随机森林或 Pipeline 中的特征重要性。"""
    actual_model = model

    if hasattr(model, "named_steps"):
        actual_model = model.named_steps.get("model", model)

    if not hasattr(actual_model, "feature_importances_"):
        st.info("当前模型不支持直接展示树模型特征重要性。")
        return

    importances = np.asarray(actual_model.feature_importances_)
    importance_df = pd.DataFrame({
        "特征": [FEATURE_NAMES.get(f, f) for f in FEATURES],
        "重要性": importances,
    }).sort_values("重要性", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        importance_df["特征"],
        importance_df["重要性"],
        color="#4C78A8",
    )
    ax.set_title("模型特征重要性")
    ax.set_xlabel("重要性")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("查看特征重要性明细"):
        st.dataframe(
            importance_df.sort_values("重要性", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def make_download_button(df, filename, label="下载 CSV"):
    content = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label=label,
        data=content,
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# 五、学业预警看板
# ============================================================

def dashboard_page(df, raw_df, model, config, model_source,
                   high_threshold, medium_threshold):
    st.header("📊 学业预警看板")

    total = len(df)
    high_count = int((df["risk_level"] == "高风险").sum())
    medium_count = int((df["risk_level"] == "中风险").sum())
    low_count = int((df["risk_level"] == "低风险").sum())

    # 第一行：核心指标
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("学生总数", f"{total:,}")
    c2.metric(
        "高风险人数",
        f"{high_count:,}",
        f"{high_count / total * 100:.1f}%" if total else "0%",
    )
    c3.metric(
        "中风险人数",
        f"{medium_count:,}",
        f"{medium_count / total * 100:.1f}%" if total else "0%",
    )
    c4.metric(
        "低风险人数",
        f"{low_count:,}",
        f"{low_count / total * 100:.1f}%" if total else "0%",
    )

    st.divider()

    st.subheader("风险等级分布")

    left, right = st.columns([1, 1])

    with left:
        show_risk_distribution(df)

    with right:
        risk_summary = pd.DataFrame({
            "风险等级": ["高风险", "中风险", "低风险"],
            "学生人数": [high_count, medium_count, low_count],
        })
        risk_summary["占比"] = (
            risk_summary["学生人数"] / total * 100
        ).round(2) if total else 0

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"当前高风险阈值：{high_threshold:.3f}；"
            f"中风险阈值：{medium_threshold:.3f}"
        )

    # 明确区分真实标签与模型预测
    st.subheader("原始标签与模型预测对照")

    original_high = int((raw_df[TARGET] == 1).sum())
    original_low = int((raw_df[TARGET] == 0).sum())

    label_col1, label_col2, label_col3 = st.columns(3)

    label_col1.metric("原始标签高风险", f"{original_high:,}")
    label_col2.metric("原始标签非高风险", f"{original_low:,}")
    label_col3.metric(
        "模型预测高风险",
        f"{high_count:,}",
        help="模型根据特征和风险阈值预测出的高风险人数，不等于原始标签人数。",
    )

    st.info(
        "说明：原始标签 at_risk=1/0 是数据集的标记；"
        "高、中、低风险是模型概率按阈值划分的结果。"
        "两者统计口径不同，不能直接视为同一个指标。"
    )

    st.divider()
    st.subheader("📌 风险因素分析")

    show_feature_importance(model)

    # 高风险学生名单
    st.divider()
    st.subheader("🚨 高风险学生名单")

    high_df = df[df["risk_level"] == "高风险"].copy()

    if high_df.empty:
        st.success("当前阈值下没有被识别为高风险的学生。")
    else:
        search_id = st.text_input(
            "按学号或学生编号搜索",
            key="dashboard_student_search",
        )

        if search_id.strip():
            high_df = high_df[
                high_df["student_id"].astype(str).str.contains(
                    search_id.strip(), case=False, na=False
                )
            ]

        display_cols = [
            "student_id",
            "risk_probability_pct",
            "attendance",
            "homework_rate",
            "study_hours",
            "past_fail",
            "risk_advice",
        ]

        display_df = high_df[display_cols].rename(columns={
            "student_id": "学生编号",
            "risk_probability_pct": "预测风险概率（%）",
            "attendance": "出勤率",
            "homework_rate": "作业完成率",
            "study_hours": "学习时长",
            "past_fail": "以往挂科情况",
            "risk_advice": "辅助建议",
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        make_download_button(
            display_df,
            "高风险学生名单.csv",
            "下载高风险学生名单",
        )

    with st.expander("查看模型与阈值信息"):
        st.write("模型状态：", model_source)
        st.write("当前配置：")
        st.json(config)


# ============================================================
# 六、批量预警
# ============================================================

def batch_prediction_page(model, high_threshold, medium_threshold):
    st.header("📁 批量学生预警")

    st.write(
        "上传 CSV 文件，系统将根据六个学生特征预测风险等级。"
    )

    st.caption(
        "所需字段：attendance、homework_rate、study_hours、"
        "past_fail、online_hours、family_support。"
    )

    uploaded_file = st.file_uploader(
        "上传学生 CSV 文件",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is None:
        st.info("请上传文件后开始批量预测。")
        return

    try:
        batch_df = pd.read_csv(uploaded_file)

        if len(batch_df.columns) == 1 and "\t" in str(batch_df.columns[0]):
            uploaded_file.seek(0)
            batch_df = pd.read_csv(uploaded_file, sep="\t")

        batch_df.columns = batch_df.columns.astype(str).str.strip()

    except Exception as exc:
        st.error(f"文件读取失败：{exc}")
        return

    missing = [col for col in FEATURES if col not in batch_df.columns]

    if missing:
        st.error(f"上传文件缺少以下字段：{', '.join(missing)}")
        return

    for col in FEATURES:
        batch_df[col] = pd.to_numeric(
            batch_df[col], errors="coerce"
        )

    batch_df[FEATURES] = batch_df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    if "student_id" not in batch_df.columns:
        batch_df.insert(
            0,
            "student_id",
            [f"BATCH{i + 1:04d}" for i in range(len(batch_df))]
        )

    try:
        probabilities = get_positive_probability(
            model, batch_df[FEATURES]
        )
    except Exception as exc:
        st.error(f"批量预测失败：{exc}")
        return

    batch_df["risk_probability"] = probabilities
    batch_df["risk_probability_pct"] = (
        batch_df["risk_probability"] * 100
    ).round(2)

    batch_df["risk_level"] = batch_df["risk_probability"].apply(
        lambda p: classify_risk(
            p, high_threshold, medium_threshold
        )
    )

    batch_df["risk_advice"] = batch_df.apply(
        lambda row: risk_advice(row, row["risk_level"]),
        axis=1,
    )

    high_count = int((batch_df["risk_level"] == "高风险").sum())
    medium_count = int((batch_df["risk_level"] == "中风险").sum())
    low_count = int((batch_df["risk_level"] == "低风险").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("高风险", high_count)
    c2.metric("中风险", medium_count)
    c3.metric("低风险", low_count)

    display_df = batch_df.rename(columns={
        "student_id": "学生编号",
        "risk_probability_pct": "预测风险概率（%）",
        "risk_level": "风险等级",
        "risk_advice": "辅助建议",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    make_download_button(
        display_df,
        "批量学业预警结果.csv",
        "下载批量预警结果",
    )


# ============================================================
# 七、单个学生预警
# ============================================================

def single_prediction_page(model, df, high_threshold, medium_threshold):
    st.header("👤 单个学生预警")

    st.write("输入学生的学习情况，查看模型预测结果和辅助建议。")

    student_id = st.text_input(
        "学生编号（可选）",
        value="NEW_STUDENT",
    )

    input_values = {}

    for feature in FEATURES:
        series = pd.to_numeric(df[feature], errors="coerce").dropna()

        if series.empty:
            minimum, maximum, median = 0.0, 100.0, 0.0
        else:
            minimum = float(series.min())
            maximum = float(series.max())
            median = float(series.median())

        if minimum == maximum:
            maximum = minimum + 1.0

        default = float(np.clip(median, minimum, maximum))

        step = 1.0 if float(minimum).is_integer() and float(maximum).is_integer() else 0.1

        input_values[feature] = st.number_input(
            FEATURE_NAMES[feature],
            min_value=float(minimum),
            max_value=float(maximum),
            value=default,
            step=step,
            key=f"single_{feature}",
        )

    if st.button("开始预警", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_values], columns=FEATURES)

        try:
            probability = float(
                get_positive_probability(model, input_df)[0]
            )
        except Exception as exc:
            st.error(f"预测失败：{exc}")
            return

        level = classify_risk(
            probability, high_threshold, medium_threshold
        )

        st.divider()
        st.subheader(f"学生编号：{student_id or 'NEW_STUDENT'}")

        c1, c2 = st.columns(2)
        c1.metric("预测风险概率", f"{probability * 100:.2f}%")
        c2.metric("风险等级", level)

        if level == "高风险":
            st.error("模型识别为高风险，建议优先进行人工核实和学业支持。")
        elif level == "中风险":
            st.warning("模型识别为中风险，建议持续观察并定期复查。")
        else:
            st.success("模型识别为低风险，建议维持常规学习支持。")

        advice = risk_advice(input_values, level)
        st.write("**辅助建议：**")
        st.write(advice)

        result = pd.DataFrame([{
            "student_id": student_id or "NEW_STUDENT",
            **input_values,
            "risk_probability": probability,
            "risk_probability_pct": round(probability * 100, 2),
            "risk_level": level,
            "risk_advice": advice,
            "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])

        make_download_button(
            result,
            "单个学生预警结果.csv",
            "下载该学生的预警结果",
        )


# ============================================================
# 八、干预记录
# ============================================================

def load_intervention_records():
    if not os.path.exists(INTERVENTION_PATH):
        return pd.DataFrame(columns=[
            "student_id",
            "risk_level",
            "intervention",
            "responsible_person",
            "follow_up_date",
            "status",
            "created_at",
        ])

    try:
        records = pd.read_csv(INTERVENTION_PATH)
        return records
    except Exception:
        return pd.DataFrame(columns=[
            "student_id",
            "risk_level",
            "intervention",
            "responsible_person",
            "follow_up_date",
            "status",
            "created_at",
        ])


def intervention_page(df):
    st.header("📝 干预记录管理")

    st.write("记录学生学业帮扶措施、负责人和后续跟进情况。")

    with st.form("intervention_form"):
        student_id = st.text_input("学生编号")
        risk_level = st.selectbox(
            "风险等级",
            ["高风险", "中风险", "低风险"],
        )
        intervention = st.text_area(
            "干预措施",
            placeholder="例如：联系学生了解情况、安排课程辅导、制定学习计划……",
        )
        responsible_person = st.text_input("负责人")
        follow_up_date = st.date_input("计划复查日期")
        status = st.selectbox(
            "处理状态",
            ["待跟进", "跟进中", "已完成"],
        )

        submitted = st.form_submit_button(
            "保存干预记录",
            type="primary",
        )

    if submitted:
        if not student_id.strip():
            st.error("请填写学生编号。")
        elif not intervention.strip():
            st.error("请填写干预措施。")
        else:
            records = load_intervention_records()

            new_record = pd.DataFrame([{
                "student_id": student_id.strip(),
                "risk_level": risk_level,
                "intervention": intervention.strip(),
                "responsible_person": responsible_person.strip(),
                "follow_up_date": str(follow_up_date),
                "status": status,
                "created_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }])

            records = pd.concat(
                [records, new_record],
                ignore_index=True,
            )

            records.to_csv(
                INTERVENTION_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            st.success("干预记录已保存。")
            st.rerun()

    st.divider()
    st.subheader("历史干预记录")

    records = load_intervention_records()

    if records.empty:
        st.info("目前还没有干预记录。")
        return

    if "status" in records.columns:
        status_filter = st.multiselect(
            "按处理状态筛选",
            options=sorted(records["status"].dropna().unique().tolist()),
            default=sorted(records["status"].dropna().unique().tolist()),
        )

        records = records[records["status"].isin(status_filter)]

    st.dataframe(
        records,
        use_container_width=True,
        hide_index=True,
    )

    make_download_button(
        records,
        "学生干预记录.csv",
        "下载干预记录",
    )


# ============================================================
# 九、实验报告
# ============================================================

def experiment_page():
    st.header("🧪 模型实验报告")

    st.write(
        "此页面展示独立实验脚本生成的模型指标、特征重要性和 SHAP 图表。"
    )

    metrics_path = os.path.join(
        OUTPUT_DIR, "final_model_metrics.csv"
    )
    importance_path = os.path.join(
        OUTPUT_DIR, "final_feature_importance.csv"
    )

    if os.path.exists(metrics_path):
        st.subheader("模型评价指标")
        try:
            metrics = pd.read_csv(metrics_path)
            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.warning(f"无法读取模型指标：{exc}")
    else:
        st.info(
            "未找到 output/final_model_metrics.csv。"
            "请先运行实验脚本生成实验结果。"
        )

    if os.path.exists(importance_path):
        st.subheader("特征重要性")
        try:
            importance = pd.read_csv(importance_path)
            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.warning(f"无法读取特征重要性：{exc}")

    image_files = [
        ("决策树深度实验", "exp1_tree_depth.png"),
        ("决策树召回率与 F1 实验", "exp1_tree_recall_f1.png"),
        ("随机森林树数量实验", "exp2_rf_estimators.png"),
        ("随机森林召回率与 F1 实验", "exp2_rf_recall_f1.png"),
        ("SHAP 特征影响分析", "shap_summary.png"),
        ("SHAP 特征重要性", "shap_feature_importance.png"),
        ("SHAP 单个学生解释", "shap_waterfall.png"),
        ("最终模型特征重要性", "final_feature_importance.png"),
    ]

    st.subheader("实验图表")

    found_image = False

    for title, filename in image_files:
        image_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(image_path):
            found_image = True
            with st.expander(title, expanded=False):
                st.image(
                    image_path,
                    caption=filename,
                    use_container_width=True,
                )

    if not found_image:
        st.info(
            "output 文件夹中暂时没有实验图表。"
            "请运行 experiment.py 后再查看。"
        )

    if os.path.isdir(OUTPUT_DIR):
        st.subheader("实验结果文件下载")

        for filename in sorted(os.listdir(OUTPUT_DIR)):
            path = os.path.join(OUTPUT_DIR, filename)

            if os.path.isfile(path) and filename.lower().endswith(".csv"):
                try:
                    with open(path, "rb") as file:
                        st.download_button(
                            label=f"下载 {filename}",
                            data=file.read(),
                            file_name=filename,
                            mime="text/csv",
                            key=f"download_{filename}",
                        )
                except OSError:
                    pass

    st.caption(
        "研究限制：如果数据由程序模拟生成，实验指标只能反映当前数据与标签规则下的效果，"
        "不能直接证明模型在真实高校学生数据上的泛化能力。"
    )


# ============================================================
# 十、主程序
# ============================================================

def main():
    show_header()

    # 加载数据
    raw_df = load_student_data()

    # 加载模型
    try:
        model, config, model_source = load_model_and_config(
            DATA_PATH,
            MODEL_PATH,
            CONFIG_PATH,
        )
    except Exception as exc:
        st.error(f"模型初始化失败：{exc}")
        st.info(
            "请检查 data/students.csv 的字段和标签，"
            "或运行正式训练脚本生成模型文件。"
        )
        st.stop()

    high_threshold, medium_threshold = get_risk_thresholds(config)

    # 预测当前数据
    try:
        signature = (
            f"{len(raw_df)}_"
            f"{os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0}_"
            f"{model_source}"
        )

        probabilities = get_model_predictions(
            signature,
            model,
            raw_df,
        )

        df = prepare_data(
            raw_df,
            probabilities,
            high_threshold,
            medium_threshold,
        )

    except Exception as exc:
        st.error(f"生成学生风险预测失败：{exc}")
        st.stop()

    with st.sidebar:
        st.title("📚 功能导航")

        page = st.radio(
            "选择功能",
            [
                "学业预警看板",
                "批量学生预警",
                "单个学生预警",
                "干预记录管理",
                "实验报告",
            ],
        )

        st.divider()

        st.caption("当前数据")
        st.write(f"学生记录：{len(df)} 条")

        st.caption("模型状态")
        st.write(model_source)

        st.caption("风险阈值")
        st.write(f"高风险：≥ {high_threshold:.3f}")
        st.write(f"中风险：≥ {medium_threshold:.3f}")

        if "备用模型" in model_source:
            st.warning(
                "当前使用备用模型。建议运行正式训练脚本，"
                "确保网页使用最终评估并保存的模型。"
            )

    if page == "学业预警看板":
        dashboard_page(
            df,
            raw_df,
            model,
            config,
            model_source,
            high_threshold,
            medium_threshold,
        )

    elif page == "批量学生预警":
        batch_prediction_page(
            model,
            high_threshold,
            medium_threshold,
        )

    elif page == "单个学生预警":
        single_prediction_page(
            model,
            df,
            high_threshold,
            medium_threshold,
        )

    elif page == "干预记录管理":
        intervention_page(df)

    elif page == "实验报告":
        experiment_page()


if __name__ == "__main__":
    main()
