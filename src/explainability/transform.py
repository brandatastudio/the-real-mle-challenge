from flask import Flask, jsonify, request, make_response
import sys

sys.path.append("/src/src")
import prediction_app.transform as pr_app_t
import yaml
import requests
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns


def plot_feature_importance(winning_model_details: dict):
    importances = winning_model_details["trained_model"].feature_importances_
    features = winning_model_details["trained_model"].feature_names_in_
    indices = np.argsort(importances)[::-1]
    importances = importances[indices]
    features = features[indices]

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.barh(range(len(importances)), importances)
    plt.yticks(range(len(importances)), features, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance", fontsize=12)

    fig.savefig("explainability/plots/feature_importance")

    return ()


def plot_confusion_matrix(winning_model_details: dict, config: dict):
    '''Function to calculate confusion matrix of different target_variable categories, they can be modified
    in the config.yaml in data_prep section, "tar_variable_category_mapping"'''

    classes = [
        int(float(x)) for x in [*config["data_prep"]["tar_variable_category_mapping"]]
    ]

    labels = [x for x in config["data_prep"]["tar_variable_category_mapping"].values()]

    conf = confusion_matrix(
        winning_model_details["y_test"], winning_model_details["y_pred"]
    )
    conf = conf / conf.sum(axis=1).reshape(len(classes), 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        conf,
        annot=True,
        cmap="BuGn",
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        cbar=False,
    )
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Real", fontsize=16)
    plt.xticks(
        ticks=np.arange(0.5, len(classes)), labels=labels, rotation=0, fontsize=12
    )
    plt.yticks(
        ticks=np.arange(0.5, len(classes)), labels=labels, rotation=0, fontsize=12
    )
    plt.title("Simple model", fontsize=18)

    fig.savefig("explainability/plots/confusion_matrix")
    return ()


def metric_evaluation_plots(winning_model_details: dict, config: dict):
    """Creates a barplot of the different metrics we track in our ml model training"""
    maps = config["data_prep"]["tar_variable_category_mapping"]
    report = winning_model_details["eval_scores"]["classification_report"]
    df_report = pd.DataFrame.from_dict(report).T[:-3]

    df_report.index = [maps[int(i)] for i in df_report.index]
    metrics = ["precision", "recall", "support"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 7))
    for i, ax in enumerate(axes):
        ax.barh(df_report.index, df_report[metrics[i]], alpha=0.9)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlabel(metrics[i], fontsize=12)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Simple model", fontsize=14)
    fig.savefig("explainability/plots/metric_by_category_barplots")
    return ()


def plot_histogram(
    data: pd.DataFrame, winning_model_details: dict, column_to_plot: str
):
    """Can plot a histogram of any column in training data set, or even the predictions, common use case is to plot target variable

    ARGS:

    data(pd.DataFrame):
    winning_model_details(dict):
    column_to_plot(str): Any column name specified in the config in 'columns_for_data_prep', any of the 'target_variable' labels
                        'y_pred' also is available argument to plot the predictions"""

    fontsize_labels = 12

    fig, ax = plt.subplots(figsize=(12, 6))
    if column_to_plot != "y_pred":
        ax.hist(data[column_to_plot], bins=range(0, max(data[column_to_plot]), 10))
    else:
        ax.hist(
            winning_model_details[column_to_plot],
            bins=range(0, max(winning_model_details[column_to_plot]), 10),
        )
    ax.grid(alpha=0.2)
    ax.set_title(column_to_plot + " distribution", fontsize=fontsize_labels)
    fig.savefig("explainability/plots/histogram")

    return ()


def plot_neighbourhood(data: pd.DataFrame):
    """Create a barplot of price by neighbourhood"""

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    axes = [ax1, ax2, ax3, ax4, ax5]
    fontsize_labels = 12

    MAP_NEIGHB = {
        1: "Bronx",
        2: "Queens",
        3: "Staten Island",
        4: "Brooklyn",
        5: "Manhattan",
    }
    data["neighbourhood"] = data["neighbourhood"].map(MAP_NEIGHB)
    neighbourhood = list(data["neighbourhood"].unique())

    for i, ax in enumerate(axes):
        values = data[data["neighbourhood"] == neighbourhood[i]]["price"]
        avg = round(values.mean(), 1)
        ax.hist(values, bins=range(0, max(data["price"]), 20))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{neighbourhood[i]}. Avg price: ${avg}", fontsize=fontsize_labels)
        ax.set_ylabel("Count", fontsize=fontsize_labels)

    ax.set_xlabel("Price ($)", fontsize=fontsize_labels)
    plt.tight_layout()
    fig.savefig("explainability/plots/histogram_by_neighbourhood")
    return ()
