
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
# 由仓库根目录的 packages.txt 安装 fonts-noto-cjk，
# 然后在运行时显式注册字体，避免 Streamlit Cloud 中文变方框。
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
    # 找不到字体时保留默认字体，避免字体问题直接导致应用启动失败；
    # 此时请检查仓库根目录 packages.txt 是否已提交并触发重新部署。
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
