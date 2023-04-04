import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import subprocess
import pdb


def split_data_for_cross_validation(
    data: pd.DataFrame, config: dict, test_size: float, random_state
):
    """Returns a list of series or array objects, with training and test sets for feature(x) and target(y) variables"""

    x = data[config["columns_for_training"]]
    y = data[config["target_variable"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    return (x_train, x_test, y_train, y_test)


def eval_metric_calculations(y_test, y_pred, y_prob) -> dict:
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True)

    metric_dictionary = {
        "acuracy": accuracy,
        "roc__auc_score": roc_auc,
        "classification_report": report,
    }

    return metric_dictionary


def gs_train_rf_classifier(self) -> dict:
    """Train random forest classifier , can only be launched as a method from a training() class object"""

    run_names = list(self.config["training_experiment_runs"].keys())

    model_run_dict = {}
    evaluation_scores_dict = {}

    for i in run_names:
        training_paramaters_dict = self.config["training_experiment_runs"][i]
        clf = RandomForestClassifier(
            n_estimators=training_paramaters_dict["n_estimators"],
            random_state=training_paramaters_dict["random_state"],
            class_weight=training_paramaters_dict["class_weight"],
            n_jobs=training_paramaters_dict["n_jobs"],
            criterion=training_paramaters_dict["criterion"],
        )
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        y_prob = clf.predict_proba(self.x_test)
        metric_dictionary = eval_metric_calculations(self.y_test, y_pred, y_prob)
        model_details = {
            "training_run_parameters": training_paramaters_dict,
            "trained_model": clf,
            "eval_scores": metric_dictionary,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_test": self.y_test,
        }
        model_run_dict[i] = model_details

        # insert model in dict and their scores
    return model_run_dict
    # metric_evaluation_process:


def pick_winning_rf_classifier(train_run_dict: dict):
    """A function that formalizes the criteria to pick a winning ml model, since this would usually
    obey business rules, and this is just an excercise, the winning model in this case will just be
    hardcoded as the one in the original excercise notebooks, in a real project, a carefull criteria would
    need to be defined based on business needs and coded here"""

    winning_model = train_run_dict["run_1"]["trained_model"]
    winning_model_details = train_run_dict["run_1"]

    return (winning_model, winning_model_details)


def gs_rf_classifiers_to_mlflow(ml_train_run_dict: dict, config: dict):
    """needs a train_run_dict result of using gs_train_rf_classifier in a train() object"""

    run_names = run_names = list(config["training_experiment_runs"].keys())

    experiment_name = config["training_experiment_name"]
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    # experiment_id = mlflow.create_experiment(config['training_experiment_name'])

    # pdb.set_trace()

    for i in run_names:
        with mlflow.start_run(run_name=i) as run:
            mlflow.log_params(config["training_experiment_runs"][i])
            mlflow.sklearn.log_model(
                ml_train_run_dict[i]["trained_model"], "classifier"
            )
            mlflow.log_dict(
                dictionary=ml_train_run_dict[i]["eval_scores"],
                artifact_file="metrics.json",
            )

            mlflow.log_artifact(config["dvc_lock_info"])

            if config["extra_log_githash"] == True:
                mlflow.log_param("git_hash", ml_train_run_dict["git_hash"])

            mlflow.end_run()

    return None


def get_current_git_commit_id() -> str:
    """this function will obtain current git commit id , this information is usefull because we can later log this in mlflow to
    facilitate reproducibility of the runs in cases where mlflow is not automatically tracking this information
    """

    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
