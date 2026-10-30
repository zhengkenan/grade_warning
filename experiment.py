# -*- coding: utf-8 -*-
"""
学生学业成绩预警系统：模型参数实验与 SHAP 可解释性分析

运行方式：
    python experiment.py

输入：
    data/students.csv

输出：
    output/exp1_tree_depth.png
    output/exp1_tree_recall_f1.png
    output/exp2_rf_estimators.png
    output/exp2_rf_recall_f1.png
    output/shap_summary.png
    output/shap_feature_importance.png
    output/shap_waterfall.png
    output/final_feature_importance.png
    output/exp1_tree_depth_results.csv
    output/exp2_rf_estimators_results.csv
    output/final_model_metrics.csv
    output/final_feature_importance.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. 路径、特征和绘图设置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "attendance",
    "homework_rate",
    "study_hours",
    "past_fail",
    "online_hours",
    "family_support"
]

TARGET = "at_risk"

FEATURE_LABELS = {
    "attendance": "出勤率",
    "homework_rate": "作业完成率",
    "study_hours": "学习时长",
    "past_fail": "以往挂科情况",
    "online_hours": "在线学习时长",
    "family_support": "家庭支持"
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def output_path(filename):
    """生成 output 目录下的文件路径。"""
    return os.path.join(OUTPUT_DIR, filename)


def save_figure(fig, filename):
    """保存图片并关闭画布，避免连续绘图时占用内存。"""
    fig.tight_layout()
    fig.savefig(output_path(filename), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成图片：output/{filename}")


# ============================================================
# 2. 读取和检查数据
# ============================================================

def load_data():
    """读取学生数据，只保留建模所需的六个特征和目标标签。"""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}\n"
            "请确认 students.csv 位于项目的 data 文件夹中。"
        )

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"数据文件缺少以下字段：{missing_columns}\n"
            f"当前字段：{df.columns.tolist()}"
        )

    # 只保留模型需要的列，学号等标识列不参与训练
    df = df[required_columns].copy()

    # 无效字符转为缺失值，由 Pipeline 在训练过程中填补
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 目标标签无法填补，必须删除
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()

    if len(df) < 10:
        raise ValueError("有效数据不足 10 条，无法可靠开展实验。")

    if df[TARGET].nunique() != 2:
        raise ValueError("at_risk 必须同时包含 0 和 1 两个类别。")

    if df[TARGET].value_counts().min() < 2:
        raise ValueError("风险类别中至少有一类不足 2 条，无法分层划分数据。")

    all_missing = df[FEATURES].isna().all()
    if all_missing.any():
        raise ValueError(
            f"以下特征列没有任何有效数值："
            f"{all_missing[all_missing].index.tolist()}"
        )

    return df.reset_index(drop=True)


# ============================================================
# 3. 建立模型与交叉验证
# ============================================================

def make_tree(max_depth):
    """决策树 Pipeline：缺失值填补只在训练折内拟合。"""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        ))
    ])


def make_forest(n_estimators=100):
    """随机森林 Pipeline。"""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])


def make_cv(y_train):
    """根据最少类别样本数确定分层交叉验证折数。"""

    min_class_count = int(pd.Series(y_train).value_counts().min())
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise ValueError("训练集类别样本不足，无法进行分层交叉验证。")

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )


def run_parameter_experiments(X_train, y_train, cv):
    """
    在训练集上进行参数实验。

    使用交叉验证评估不同参数，不使用独立测试集选择参数，
    避免反复根据测试集结果调参。
    """

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auc": "roc_auc"
    }

    # ---------------- 决策树深度实验 ----------------

    tree_depths = list(range(2, 16))
    tree_rows = []

    for depth in tree_depths:
        model = make_tree(depth)

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        tree_rows.append({
            "max_depth": depth,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "auc": scores["test_auc"].mean()
        })

    tree_df = pd.DataFrame(tree_rows)
    tree_df.to_csv(
        output_path("exp1_tree_depth_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 决策树深度与交叉验证 AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tree_df["max_depth"],
        tree_df["auc"],
        marker="o",
        label="交叉验证 AUC"
    )
    ax.set_xlabel("决策树最大深度")
    ax.set_ylabel("交叉验证 AUC")
    ax.set_title("决策树不同深度的性能变化")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "exp1_tree_depth.png")

    # 决策树召回率与 F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tree_df["max_depth"],
        tree_df["recall"],
        marker="o",
        label="召回率"
    )
    ax.plot(
        tree_df["max_depth"],
        tree_df["f1"],
        marker="s",
        label="F1"
    )
    ax.set_xlabel("决策树最大深度")
    ax.set_ylabel("交叉验证指标")
    ax.set_ylim(0, 1.05)
    ax.set_title("决策树深度对召回率与 F1 的影响")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "exp1_tree_recall_f1.png")

    # ---------------- 随机森林树数量实验 ----------------

    estimator_counts = [10, 30, 50, 80, 100, 150, 200]
    forest_rows = []

    for n_estimators in estimator_counts:
        model = make_forest(n_estimators)

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        forest_rows.append({
            "n_estimators": n_estimators,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "auc": scores["test_auc"].mean()
        })

    forest_df = pd.DataFrame(forest_rows)
    forest_df.to_csv(
        output_path("exp2_rf_estimators_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 随机森林树数量与 AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        forest_df["n_estimators"],
        forest_df["auc"],
        marker="o",
        label="交叉验证 AUC"
    )
    ax.set_xlabel("随机森林决策树数量")
    ax.set_ylabel("交叉验证 AUC")
    ax.set_title("随机森林不同树数量的性能变化")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "exp2_rf_estimators.png")

    # 随机森林召回率与 F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        forest_df["n_estimators"],
        forest_df["recall"],
        marker="o",
        label="召回率"
    )
    ax.plot(
        forest_df["n_estimators"],
        forest_df["f1"],
        marker="s",
        label="F1"
    )
    ax.set_xlabel("随机森林决策树数量")
    ax.set_ylabel("交叉验证指标")
    ax.set_ylim(0, 1.05)
    ax.set_title("随机森林树数量对召回率与 F1 的影响")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "exp2_rf_recall_f1.png")

    print("\n决策树参数实验结果：")
    print(tree_df.round(4).to_string(index=False))

    print("\n随机森林参数实验结果：")
    print(forest_df.round(4).to_string(index=False))

    return tree_df, forest_df


# ============================================================
# 4. 最终模型评估与特征重要性
# ============================================================

def evaluate_final_model(X_train, X_test, y_train, y_test):
    """训练最终随机森林模型，并仅在独立测试集上评估一次。"""

    model = make_forest(n_estimators=100)
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "模型": "随机森林",
        "测试集准确率": accuracy_score(y_test, test_pred),
        "测试集精确率": precision_score(
            y_test, test_pred, zero_division=0
        ),
        "测试集召回率": recall_score(
            y_test, test_pred, zero_division=0
        ),
        "测试集F1": f1_score(
            y_test, test_pred, zero_division=0
        ),
        "测试集AUC": roc_auc_score(y_test, test_prob)
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        output_path("final_model_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n最终随机森林模型测试集指标：")
    print(metrics_df.round(4).to_string(index=False))

    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, test_pred))

    print("\n分类报告：")
    print(classification_report(
        y_test,
        test_pred,
        target_names=["非预警学生", "预警学生"],
        zero_division=0
    ))

    # 最终随机森林特征重要性
    forest = model.named_steps["model"]
    importance_df = pd.DataFrame({
        "特征": FEATURES,
        "特征名称": [FEATURE_LABELS[x] for x in FEATURES],
        "重要性": forest.feature_importances_
    }).sort_values("重要性", ascending=False)

    importance_df.to_csv(
        output_path("final_feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = importance_df.sort_values("重要性", ascending=True)
    ax.barh(plot_df["特征名称"], plot_df["重要性"])
    ax.set_xlabel("特征重要性")
    ax.set_title("最终随机森林特征重要性")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "final_feature_importance.png")

    return model, importance_df


# ============================================================
# 5. SHAP 可解释性分析
# ============================================================

def run_shap_analysis(model, X_train, X_test):
    """
    使用训练集拟合的随机森林模型开展 SHAP 分析。

    如果未安装 SHAP，或当前 SHAP 版本与绘图接口不兼容，
    仅跳过 SHAP 图片，不影响前面的参数实验和最终模型评估。
    """

    try:
        import shap
    except ImportError:
        print(
            "\n未检测到 shap，已跳过 SHAP 分析。"
            "\n如需启用，请在依赖文件中添加 shap。"
        )
        return

    try:
        forest = model.named_steps["model"]
        imputer = model.named_steps["imputer"]

        # 使用训练集拟合的填补器转换数据，不重新拟合
        X_train_imp = pd.DataFrame(
            imputer.transform(X_train),
            columns=FEATURES,
            index=X_train.index
        )

        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=FEATURES,
            index=X_test.index
        )

        # 控制 SHAP 计算规模，避免数据量较大时耗时过长
        background = X_train_imp.sample(
            n=min(100, len(X_train_imp)),
            random_state=42
        )

        explain_data = X_test_imp.sample(
            n=min(100, len(X_test_imp)),
            random_state=42
        )

        explainer = shap.TreeExplainer(
            forest,
            data=background,
            feature_perturbation="interventional"
        )

        shap_result = explainer(explain_data)

        values = np.asarray(shap_result.values)
        base_values = np.asarray(shap_result.base_values)

        # 新版 SHAP 的二分类输出有时是 (样本, 特征, 类别)
        if values.ndim == 3:
            values = values[:, :, 1]

            if base_values.ndim == 2:
                base_values = base_values[:, 1]

        # 兼容旧版 SHAP 的返回形式
        elif values.ndim != 2:
            old_result = explainer.shap_values(explain_data)

            if isinstance(old_result, list):
                values = np.asarray(old_result[1])
            else:
                values = np.asarray(old_result)

                if values.ndim == 3:
                    values = values[:, :, 1]

            if base_values.ndim == 2:
                base_values = base_values[:, 1]

        feature_names = [
            FEATURE_LABELS.get(name, name) for name in FEATURES
        ]

        # ---------------- SHAP 全局摘要图 ----------------

        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            values,
            explain_data,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP 全局特征影响分析")
        plt.tight_layout()
        plt.savefig(
            output_path("shap_summary.png"),
            dpi=220,
            bbox_inches="tight"
        )
        plt.close()
        print("已生成图片：output/shap_summary.png")

        # ---------------- SHAP 特征重要性图 ----------------

        mean_abs_values = np.abs(values).mean(axis=0)

        shap_importance = pd.DataFrame({
            "特征": FEATURES,
            "特征名称": feature_names,
            "平均绝对 SHAP 值": mean_abs_values
        }).sort_values("平均绝对 SHAP 值", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            shap_importance["特征名称"],
            shap_importance["平均绝对 SHAP 值"]
        )
        ax.set_xlabel("平均绝对 SHAP 值")
        ax.set_title("SHAP 特征重要性")
        ax.grid(axis="x", alpha=0.25)
        save_figure(fig, "shap_feature_importance.png")

        # ---------------- 单个学生风险解释 ----------------

        probabilities = model.predict_proba(X_test)[:, 1]
        high_risk_index = int(np.argmax(probabilities))

        # 尝试生成 SHAP 瀑布图
        try:
            if base_values.ndim == 0:
                selected_base = float(base_values)
            else:
                selected_base = float(
                    np.asarray(base_values).reshape(-1)[high_risk_index]
                )

            selected_explanation = shap.Explanation(
                values=values[high_risk_index],
                base_values=selected_base,
                data=explain_data.iloc[high_risk_index].values,
                feature_names=feature_names
            )

            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(
                selected_explanation,
                show=False,
                max_display=len(FEATURES)
            )
            plt.title(
                "单个学生风险解释"
                f"（模型预测概率：{probabilities[high_risk_index]:.3f}）"
            )
            plt.tight_layout()
            plt.savefig(
                output_path("shap_waterfall.png"),
                dpi=220,
                bbox_inches="tight"
            )
            plt.close()
            print("已生成图片：output/shap_waterfall.png")

        except Exception as waterfall_error:
            # 瀑布图接口不兼容时，仍然生成单个学生的贡献条形图
            print(f"SHAP 瀑布图生成失败，改用条形图：{waterfall_error}")

            selected_values = values[high_risk_index]
            order = np.argsort(np.abs(selected_values))

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(
                np.array(feature_names)[order],
                selected_values[order]
            )
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_xlabel("SHAP 值（正值推动模型输出上升）")
            ax.set_title(
                "单个学生特征贡献"
                f"（预测概率：{probabilities[high_risk_index]:.3f}）"
            )
            ax.grid(axis="x", alpha=0.25)
            save_figure(fig, "shap_waterfall.png")

        # 保存被解释学生的特征与 SHAP 值
        details = pd.DataFrame({
            "特征": FEATURES,
            "特征名称": feature_names,
            "特征取值": explain_data.iloc[high_risk_index].values,
            "SHAP值": values[high_risk_index]
        })

        details["模型预测预警概率"] = probabilities[high_risk_index]
        details.to_csv(
            output_path("shap_single_student_details.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        print("\nSHAP 分析完成。")

    except Exception as e:
        print("\nSHAP 分析未能完成，但其他实验结果已保留。")
        print(f"具体原因：{e}")


# ============================================================
# 6. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("学生学业成绩预警系统：模型实验开始")
    print("=" * 60)

    df = load_data()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"有效样本数：{len(df)}")
    print(f"训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")
    print(f"特征数量：{len(FEATURES)}")

    cv = make_cv(y_train)

    # 1. 参数实验只使用训练集及其交叉验证
    run_parameter_experiments(X_train, y_train, cv)

    # 2. 最终模型在独立测试集上评估
    model, importance_df = evaluate_final_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # 3. SHAP 可解释性分析
    run_shap_analysis(model, X_train, X_test)

    print("\n" + "=" * 60)
    print("实验运行结束。请检查项目 output 文件夹。")
    print("=" * 60)


if __name__ == "__main__":
    main()
