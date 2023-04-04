import sys
import os

sys.path.append("/src/src")
import data_prep.transform as dp_tr
import training.task as tr_t
import yaml
import pandas as pd
import pdb
import pickle
import utility as util


# import data_prep.transform as data_prep_t
# import data_prep.task as data_prep_task


def prepare_request_data_for_prediction(
    data: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """An auxiliary function for predict, transformations from data_prep are applied to request data to make sure
    columns in the data are ready for inference"""

    training_config = config["training"]

    data = data[training_config["columns_for_training"]]
    data = dp_tr.neighbourhood_transformations(data)
    data = dp_tr.room_type_transformations(data)

    data = data.select_dtypes(["number"])
    return data


def predict(config: dict, data_dict: dict) -> str:
    """This function will take care of receiving the get request info and returning prediction, the price category"""

    data = pd.DataFrame(data_dict, index=[0])
    data = prepare_request_data_for_prediction(data=data, config=config)

    loaded_model = pickle.load(open(config["training"]["ml_model_path"], "rb"))
    prediction = loaded_model.predict(data)[0]
    prediction_text = util.map_price_category_code_to_text(code=prediction)
    return prediction_text


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

        test_input = {
            "id": 1001,
            "accommodates": 4,
            "room_type": "Entire home/apt",
            "beds": 2,
            "bedrooms": 1,
            "bathrooms": 2,
            "neighbourhood": "Brooklyn",
            "tv": 1,
            "elevator": 1,
            "internet": 0,
            "latitude": 40.71383,
            "longitude": -73.9658,
        }

        prediction = predict(config=config, data_dict=test_input)
        print({"id": test_input["id"], "price_category": prediction})
