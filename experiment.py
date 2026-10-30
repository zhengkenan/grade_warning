
# -*- coding: utf-8 -*-
"""
学生学业成绩预警系统——模型实验与可解释性分析

实验内容：
1. 决策树深度对模型性能的影响
2. 随机森林树数量对模型性能的影响
3. 基于 SHAP 的模型可解释性分析

数据文件：data/students.csv
输出目录：output/

注意：
本项目使用模拟学生数据，实验结果用于验证系统设计与算法流程，
不能直接视为模型在真实学生群体上的泛化性能。
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
# 0. 基础配置
# ============================================================

DATA_PATH = "data/students.csv"
OUTPUT_DIR = "output"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = [
    "attendance",
    "homework_rate",
    "study_hours",
    "past_fail",
    "online_hours",
    "family_support",
]

FEATURE_LABELS = {
    "attendance": "出勤率",
    "homework_rate": "作业完成率",
    "study_hours": "学习时长",
    "past_fail": "以往挂科情况",
    "online_hours": "在线学习时长",
    "family_support": "家庭支持",
}

TARGET = "at_risk"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 中文字体设置
# ============================================================

def setup_chinese_font():
    """尝试设置常见中文字体，兼容不同操作系统。"""
    from matplotlib import font_manager

    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]

    available_fonts = {
        font.name for font in font_manager.fontManager.ttflist
    }

    for font_name in candidate_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()


# ============================================================
# 2. 数据读取与检查
# ============================================================

def load_dataset():
    """读取数据、检查字段、处理缺失值和非法标签。"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}\n"
            "请确认 students.csv 位于项目的 data 文件夹中。"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("学生数据文件为空，无法进行实验。")

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据文件缺少必要字段："
            + ", ".join(missing_columns)
            + "\n请检查 students.csv 的列名。"
        )

    # 仅保留本次实验需要的字段
    df = df[required_columns].copy()

    # 将输入特征和标签转换为数值
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 标签缺失的样本无法用于监督学习，予以删除
    before_count = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    dropped_count = before_count - len(df)

    # 标签必须为 0 或 1
    df = df[df[TARGET].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("清理后没有可用数据，请检查 at_risk 标签。")

    # 缺失值使用当前可用样本的中位数填补
    # 若某一列全部缺失，则使用 0 填补并输出警告
    for col in FEATURES:
        median_value = df[col].median()

        if pd.isna(median_value):
            warnings.warn(
                f"特征 {col} 全部缺失，将使用 0 填补。",
                RuntimeWarning,
            )
            median_value = 0

        df[col] = df[col].fillna(median_value)

    # 标签转为整数
    df[TARGET] = df[TARGET].astype(int)

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "at_risk 只有一个类别，无法训练二分类模型。"
        )

    # 检查各类别样本数是否支持分层划分
    class_counts = df[TARGET].value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "至少一个标签类别的样本数少于 2，无法进行分层训练/测试划分。"
        )

    print("=" * 60)
    print("数据集检查结果")
    print("=" * 60)
    print(f"有效样本数：{len(df)}")
    print(f"风险学生数：{int((df[TARGET] == 1).sum())}")
    print(f"非风险学生数：{int((df[TARGET] == 0).sum())}")
    print(f"删除标签缺失样本数：{dropped_count}")
    print("\n标签分布：")
    print(df[TARGET].value_counts().sort_index())

    return df


# ============================================================
# 3. 评价指标
# ============================================================

def evaluate_model(model, X_data, y_true, model_name):
    """计算二分类模型的主要评价指标。"""
    y_pred = model.predict(X_data)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_data)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    return {
        "模型": model_name,
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "召回率": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1值": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "AUC": auc,
    }


def save_figure(filename):
    """保存当前图表并关闭画布。"""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{path}")


# ============================================================
# 4. 划分训练集与测试集
# ============================================================

df = load_dataset()

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))


# ============================================================
# 实验一：决策树深度对模型性能的影响
# ============================================================

print("\n" + "=" * 60)
print("实验一：决策树深度对模型性能的影响")
print("=" * 60)

depths = list(range(2, 16))

tree_results = []

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    train_metrics = evaluate_model(
        model, X_train, y_train, f"深度={depth}（训练集）"
    )

    test_metrics = evaluate_model(
        model, X_test, y_test, f"深度={depth}（测试集）"
    )

    tree_results.append({
        "最大深度": depth,
        "训练集准确率": train_metrics["准确率"],
        "测试集准确率": test_metrics["准确率"],
        "测试集精确率": test_metrics["精确率"],
        "测试集召回率": test_metrics["召回率"],
        "测试集F1值": test_metrics["F1值"],
        "测试集AUC": test_metrics["AUC"],
    })

tree_results_df = pd.DataFrame(tree_results)

tree_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "exp1_tree_depth_results.csv"),
    index=False,
    encoding="utf-8-sig",
)

print(tree_results_df.round(4).to_string(index=False))

# 图1：训练集和测试集准确率
plt.figure(figsize=(9, 5.5))

plt.plot(
    depths,
    tree_results_df["训练集准确率"],
    marker="o",
    linewidth=2,
    label="训练集准确率",
)

plt.plot(
    depths,
    tree_results_df["测试集准确率"],
    marker="s",
    linewidth=2,
    label="测试集准确率",
)

plt.xlabel("决策树最大深度")
plt.ylabel("准确率")
plt.title("决策树深度对准确率的影响")
plt.xticks(depths)
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
save_figure("exp1_tree_depth.png")

# 图2：不同深度的召回率与 F1
plt.figure(figsize=(9, 5.5))

plt.plot(
    depths,
    tree_results_df["测试集召回率"],
    marker="o",
    linewidth=2,
    label="测试集召回率",
)

plt.plot(
    depths,
    tree_results_df["测试集F1值"],
    marker="s",
    linewidth=2,
    label="测试集F1值",
)

plt.xlabel("决策树最大深度")
plt.ylabel("评价指标")
plt.title("决策树深度对召回率和F1值的影响")
plt.xticks(depths)
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
save_figure("exp1_tree_recall_f1.png")


# ============================================================
# 实验二：随机森林树数量对模型性能的影响
# ============================================================

print("\n" + "=" * 60)
print("实验二：随机森林树数量对模型性能的影响")
print("=" * 60)

n_estimators_list = [10, 30, 50, 80, 100, 150, 200]

rf_results = []

for n_estimators in n_estimators_list:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        f"随机森林（{n_estimators}棵树）",
    )

    rf_results.append({
        "树数量": n_estimators,
        "准确率": metrics["准确率"],
        "精确率": metrics["精确率"],
        "召回率": metrics["召回率"],
        "F1值": metrics["F1值"],
        "AUC": metrics["AUC"],
    })

rf_results_df = pd.DataFrame(rf_results)

rf_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "exp2_rf_estimators_results.csv"),
    index=False,
    encoding="utf-8-sig",
)

print(rf_results_df.round(4).to_string(index=False))

# 图3：不同树数量对 AUC 的影响
plt.figure(figsize=(9, 5.5))

plt.plot(
    n_estimators_list,
    rf_results_df["AUC"],
    marker="^",
    linewidth=2,
    color="green",
)

plt.xlabel("随机森林树的数量")
plt.ylabel("AUC")
plt.title("随机森林树数量对AUC的影响")
plt.xticks(n_estimators_list)
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
save_figure("exp2_rf_estimators.png")

# 图4：不同树数量对召回率和 F1 的影响
plt.figure(figsize=(9, 5.5))

plt.plot(
    n_estimators_list,
    rf_results_df["召回率"],
    marker="o",
    linewidth=2,
    label="召回率",
)

plt.plot(
    n_estimators_list,
    rf_results_df["F1值"],
    marker="s",
    linewidth=2,
    label="F1值",
)

plt.xlabel("随机森林树的数量")
plt.ylabel("评价指标")
plt.title("随机森林树数量对召回率和F1值的影响")
plt.xticks(n_estimators_list)
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
save_figure("exp2_rf_recall_f1.png")


# ============================================================
# 实验三：SHAP 模型可解释性分析
# ============================================================

print("\n" + "=" * 60)
print("实验三：SHAP模型可解释性分析")
print("=" * 60)

try:
    import shap

    # 使用训练集训练解释模型，避免用测试集拟合模型
    shap_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    shap_model.fit(X_train, y_train)

    # 选择测试集中预测风险概率最高的学生
    test_probabilities = shap_model.predict_proba(X_test)[:, 1]
    high_risk_position = int(np.argmax(test_probabilities))

    high_risk_student = X_test.iloc[[high_risk_position]]
    high_risk_probability = float(
        test_probabilities[high_risk_position]
    )

    print(
        "用于解释的测试样本风险预测概率："
        f"{high_risk_probability:.4f}"
    )

    # 使用新版 SHAP Explanation 接口
    explainer = shap.TreeExplainer(shap_model)
    shap_explanation = explainer(X_test)

    # 获取风险类别（类别 1）的 SHAP 值
    # 兼容常见的二维和三维 Explanation 数组结构
    values = np.asarray(shap_explanation.values)

    if values.ndim == 3:
        risk_class_index = list(shap_model.classes_).index(1)
        risk_values = values[:, :, risk_class_index]
        base_values = np.asarray(shap_explanation.base_values)

        if base_values.ndim == 2:
            risk_base_values = base_values[:, risk_class_index]
        else:
            risk_base_values = base_values

        risk_explanation = shap.Explanation(
            values=risk_values,
            base_values=risk_base_values,
            data=X_test.to_numpy(),
            feature_names=FEATURES,
        )

    elif values.ndim == 2:
        risk_explanation = shap_explanation

    else:
        raise ValueError(
            f"不支持的 SHAP 值维度：{values.shape}"
        )

    # 图5：SHAP 蜂群图
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        risk_explanation.values,
        X_test,
        feature_names=[
            FEATURE_LABELS.get(feature, feature)
            for feature in FEATURES
        ],
        show=False,
    )

    save_figure("shap_summary.png")

    # 图6：SHAP 特征重要性条形图
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        risk_explanation.values,
        X_test.rename(columns=FEATURE_LABELS),
        plot_type="bar",
        show=False,
    )

    save_figure("shap_feature_importance.png")

    # 图7：单个高风险学生瀑布图
    # 这里的索引是测试集中的位置，而非原始 DataFrame 的行号
    single_explanation = risk_explanation[high_risk_position]

    plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        single_explanation,
        max_display=len(FEATURES),
        show=False,
    )

    save_figure("shap_waterfall.png")

    # 导出该学生的特征值、SHAP 值和风险概率
    single_shap_values = np.asarray(single_explanation.values).reshape(-1)

    shap_detail_df = pd.DataFrame({
        "特征": [
            FEATURE_LABELS.get(feature, feature)
            for feature in FEATURES
        ],
        "特征值": high_risk_student.iloc[0].to_numpy(),
        "SHAP值": single_shap_values,
    })

    shap_detail_df["SHAP绝对值"] = shap_detail_df["SHAP值"].abs()
    shap_detail_df = shap_detail_df.sort_values(
        "SHAP绝对值",
        ascending=False,
    )

    shap_detail_df.to_csv(
        os.path.join(OUTPUT_DIR, "shap_single_student_details.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n单个学生的 SHAP 解释结果：")
    print(shap_detail_df.round(4).to_string(index=False))

    print(
        "\n说明：SHAP 值反映特征对当前模型输出的贡献，"
        "不代表特征与学业风险之间存在因果关系。"
    )

except ImportError:
    print(
        "\n未安装 shap，已跳过 SHAP 实验。"
        "\n请在 requirements.txt 中添加 shap，"
        "并重新部署后再运行。"
    )

except Exception as exc:
    print("\nSHAP 实验执行失败：", repr(exc))
    print(
        "请检查 SHAP 与 NumPy 的版本兼容性，"
        "以及当前数据和模型的输出格式。"
    )


# ============================================================
# 5. 最终模型评估与报告导出
# ============================================================

print("\n" + "=" * 60)
print("最终随机森林模型评估")
print("=" * 60)

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_model.fit(X_train, y_train)

final_metrics = evaluate_model(
    final_model,
    X_test,
    y_test,
    "随机森林最终模型",
)

for metric_name, metric_value in final_metrics.items():
    if isinstance(metric_value, (int, float, np.integer, np.floating)):
        if pd.isna(metric_value):
            print(f"{metric_name}：无法计算")
        else:
            print(f"{metric_name}：{metric_value:.4f}")
    else:
        print(f"{metric_name}：{metric_value}")

y_pred = final_model.predict(X_test)

print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\n分类报告：")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["非风险学生", "风险学生"],
        zero_division=0,
    )
)

# 保存最终模型评估指标
pd.DataFrame([final_metrics]).to_csv(
    os.path.join(OUTPUT_DIR, "final_model_metrics.csv"),
    index=False,
    encoding="utf-8-sig",
)

# 保存最终模型特征重要性
importance_df = pd.DataFrame({
    "特征": [
        FEATURE_LABELS.get(feature, feature)
        for feature in FEATURES
    ],
    "特征名称": FEATURES,
    "重要性": final_model.feature_importances_,
}).sort_values("重要性", ascending=False)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_feature_importance.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\n最终模型特征重要性：")
print(importance_df.round(4).to_string(index=False))

# 图8：最终模型特征重要性
plt.figure(figsize=(9, 5.5))

plot_importance = importance_df.sort_values(
    "重要性",
    ascending=True,
)

plt.barh(
    plot_importance["特征"],
    plot_importance["重要性"],
)

plt.xlabel("特征重要性")
plt.ylabel("学生特征")
plt.title("随机森林特征重要性分析")
plt.grid(axis="x", alpha=0.25)

save_figure("final_feature_importance.png")


# ============================================================
# 6. 完成提示
# ============================================================

print("\n" + "=" * 60)
print("实验执行结束")
print("=" * 60)
print(f"所有图表和实验结果已输出至：{OUTPUT_DIR}")
print("本次实验包含：")
print("1. 决策树深度实验")
print("2. 随机森林树数量实验")
print("3. SHAP 可解释性实验（若依赖安装成功）")
print("4. 最终模型评价指标、分类报告和特征重要性")
print("\n请在论文中说明数据集为模拟数据，并谨慎解释实验结果。")
