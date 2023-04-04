import pandas as pd
import sys
from time import gmtime, strftime

sys.path.append("/src/src")
import data_prep.transform as dp_t
import numpy as np
import yaml


class read:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_raw_data(self, sep: str = ",") -> pd.DataFrame:
        """reads csv raw data and generates pandas df"""

        data = pd.read_csv(filepath_or_buffer=self.data_path, sep=sep)
        return data


class data_prep:
    def __init__(self, raw_data, config):
        self.raw_data = raw_data
        self.config = config
        self.target_variable = self.config["target_variable"]

    def get_quality_details(self) -> dict:
        data_quality_details = dp_t.data_quality_details(self.raw_data)
        return data_quality_details

    def prepare_data(self) -> pd.DataFrame:
        processed_data = self.raw_data.copy()
        if self.target_variable in self.raw_data.columns:
            processed_data = dp_t.prepare_target_variable(
                data=processed_data,
                target_variable=self.target_variable,
                config=self.config,
            )

        processed_data = dp_t.prepare_features(data=processed_data, config=self.config)
        processed_data = processed_data.dropna(axis=0)

        return processed_data


class load:
    def __init__(self, processed_data: pd.DataFrame, config: dict, data_details: dict):
        self.processed_data = processed_data
        self.config = config
        self.data_details = data_details

    def store_locally_data(self):
        self.processed_data.to_csv(config["prepped_for_Ml_data_path"])


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)["data_prep"]

        try:
            data = read(data_path=config["raw_data_path"]).get_raw_data()
            sys.stdout.write(
                "analysis_read_log: Data properly read"
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        except Exception as e:
            sys.stderr.write(
                "analysis_read_log: data read failed |error message:"
                + str(e)
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        try:
            data_prep = data_prep(raw_data=data, config=config)
            data_quality_details = data_prep.get_quality_details()

            processed_data = data_prep.prepare_data()
            data_quality_details["target_variable_statistics"] = processed_data[
                config["target_variable"]
            ].describe()
            data_quality_details["categoric_price_count"] = processed_data[
                "price_category"
            ].count()
            sys.stdout.write(
                "data_preparation_log: Data prepared and cleaned"
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        except Exception as e:
            sys.stderr.write(
                "data_preparation_log: data preparation failed |error message:"
                + str(e)
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        try:
            load = load(
                processed_data=processed_data,
                config=config,
                data_details=data_quality_details,
            )
            load.store_locally_data()
            sys.stdout.write(
                "data_storage_log: Data successfully stored"
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )

        except Exception as e:
            sys.stderr.write(
                "data_storage_log: local data storage failed |error message:"
                + str(e)
                + "|Time of run:"
                + str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            )
