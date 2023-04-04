import pandas as pd
import sys
from time import gmtime, strftime

sys.path.append("/src/src")
import training.transform as tr_t
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import pickle


class read:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self, sep: str = ",") -> pd.DataFrame:
        """reads csv raw data and generates pandas df"""

        data = pd.read_csv(filepath_or_buffer=self.data_path, sep=sep)
        return data


class train:
    def __init__(self, data: pd.DataFrame, config: dir):
        self.config = config
        self.data = data

    def train(self):
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = tr_t.split_data_for_cross_validation(
            data=self.data,
            config=self.config,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
        )

        self.train_run_dict = tr_t.gs_train_rf_classifier(self)

        (
            self.winning_model,
            self.winning_model_details,
        ) = tr_t.pick_winning_rf_classifier(self.train_run_dict)

        # create a function to pick a winner, pick that winning model and added as an object in the train_class


class load:
    def __init__(
        self,
        config,
        ml_train_run_dict: dict,
        ml_winning_model,
        ml_winning_model_details: dict,
    ):
        self.config = config
        self.ml_train_run_dict = ml_train_run_dict
        self.ml_winning_model = ml_winning_model
        self.ml_winning_model_details = ml_winning_model_details
        self.git_hash = tr_t.get_current_git_commit_id()
        self.ml_train_run_dict["git_hash"] = self.git_hash

    def write_runs_to_mlflow(self):
        tr_t.gs_rf_classifiers_to_mlflow(
            ml_train_run_dict=self.ml_train_run_dict, config=self.config
        )

        return None

    def store_winning_ml_model(self):
        pickle.dump(self.ml_winning_model, open(config["ml_model_path"], "wb"))
        pickle.dump(
            self.ml_winning_model_details, open(config["ml_model_details_path"], "wb")
        )

        return None


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)["training"]

        try:
            data = read(data_path=config["prepped_for_Ml_data_path"]).get_data()
            sys.stdout.write(
                "training data read log: Data properly read"
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        except Exception as e:
            sys.stderr.write(
                "training data read log: data read failed |error message:"
                + str(e)
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        try:
            training = train(data=data, config=config)
            training.train()

            sys.stdout.write(
                "training  log: model properly trained"
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        except Exception as e:
            sys.stderr.write(
                "training  log: model training failed |error message:"
                + str(e)
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        try:
            load = load(
                config=config,
                ml_train_run_dict=training.train_run_dict,
                ml_winning_model=training.winning_model,
                ml_winning_model_details=training.winning_model_details,
            )

            load.write_runs_to_mlflow()

            load.store_winning_ml_model()

            sys.stdout.write(
                "loading  log: model and model info properly saved"
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        except Exception as e:
            sys.stderr.write(
                "loading log: loading process failed |error message:"
                + str(e)
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )
