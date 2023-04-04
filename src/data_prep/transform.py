import pandas as pd
import numpy as np


def data_quality_details(raw_data: pd.DataFrame) -> dict:
    """This function will get basic data quality details for the specific raw_data, and
    return the info of interest as a dictionary


    ARGS:
    raw_data(pd.DataFrame)

    RETURNS:
    quality_details(dict)"""

    data_shape = raw_data.shape
    data_columns = raw_data.columns
    data_statistics = raw_data.describe()
    data_missing_values = raw_data.isna().sum()

    quality_details = {
        "data_shape": data_shape,
        "data_columns": data_columns,
        "data_statistics": data_statistics,
        "data_missing_values": data_missing_values,
    }

    return quality_details


def remove_threshold_of_numeric_column(
    data: pd.DataFrame, numeric_column: str = "price", threshold: int = 10
) -> pd.DataFrame:
    """Function used to remove values of a numeric column representing a specific threshold,
    common use case is to remove listings of target_variable between 0 and 10"""

    data = data[data[numeric_column] >= threshold]
    return data


def get_bathroom_nums(data: pd.DataFrame, bathroom_text_column: str) -> pd.DataFrame:
    """Gets number of bathrooms from a text column
    ARGS
    data(pd.DataFrame) ,
    bathroom_column(str): the text column in data that has a string of text in the example structure: 1 bath

    RETURNS
    data(pd.Dataframe):with a new column called "bathrooms" inside
    """

    def get_nums_from_text(text):
        try:
            if isinstance(text, str):
                bath_num = text.split(" ")[0]
                return float(bath_num)
            else:
                return np.NaN
        except ValueError:
            return np.NaN

    data["bathrooms"] = data[bathroom_text_column].apply(get_nums_from_text)

    return data


def prepare_target_variable(
    data: pd.DataFrame, target_variable: str, config: dict
) -> pd.DataFrame:
    """does necesary transformations on price to generate target variable for ml_model"""

    processed_data = data.copy()

    numeric_dtypes = [np.dtype("int64"), np.dtype("float64")]
    if processed_data[target_variable].dtype is not numeric_dtypes:
        processed_data[target_variable] = (
            processed_data[target_variable].str.extract(r"(\d+).").astype(int)
        )

    processed_data = remove_threshold_of_numeric_column(
        data=processed_data, numeric_column=target_variable, threshold=10
    )

    processed_data["price_category"] = pd.cut(
        processed_data[target_variable],
        bins=[10, 90, 180, 400, np.inf],
        labels=[int(float(x)) for x in [*config["tar_variable_category_mapping"]]],
    )

    return processed_data


def binary_columns_from_text(
    data: pd.DataFrame,
    column_to_check: str,
    columns_names: list,
    columns_text_to_check: list,
) -> pd.DataFrame:
    """this function will create binary variables based on the text on a specific column,
    it will check the texts in the column_text_to_check list by index order, and will create a column
    in the same order using the names in columns_names list

    ARGS:
    data(pd.DataFrame)
    column_to_check(str): the column with text that needs to be checked
    columns_names(list): a list with the name of the binary columns that will be created
    columns_text_to_check(list): a list with the text to check for each column that will be created

    RETURNS:
    Data(pd.Dataframe): now it will include a binary column for each name in column_names, with 1 if the text
    in columns_text_to_check was found and 0 if it wasn't for each row of data

    NOTES:
    if columns_text_to_check is left as None in the config file, the text becomes the column names
    if columns_names and column_to_check are not the same size, it returns an ERROR.

    """

    if type(columns_text_to_check) != list:
        columns_text_to_check = columns_names

    if len(columns_text_to_check) != len(columns_names):
        raise ValueError(
            "columns_text_to_check and columns_names should be same size lists"
        )
        return data

    else:
        for i in range(0, len(columns_names)):
            data[columns_names[i]] = (
                data[column_to_check].str.contains(columns_text_to_check[i]).astype(int)
            )

        return data


def neighbourhood_transformations(data):
    if "neighbourhood_group_cleansed" in data.columns:
        data.rename(
            columns={"neighbourhood_group_cleansed": "neighbourhood"}, inplace=True
        )

    if "neighbourhood" in data.columns:
        MAP_NEIGHB = {
            "Bronx": 1,
            "Queens": 2,
            "Staten Island": 3,
            "Brooklyn": 4,
            "Manhattan": 5,
        }
        data["neighbourhood"] = data["neighbourhood"].map(MAP_NEIGHB)
        return data

    else:
        return print("no neigbourhood column")


def room_type_transformations(data: pd.DataFrame) -> pd.DataFrame:
    MAP_ROOM_TYPE = {
        "Shared room": 1,
        "Private room": 2,
        "Entire home/apt": 3,
        "Hotel room": 4,
    }
    data["room_type"] = data["room_type"].map(MAP_ROOM_TYPE)

    return data


def prepare_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """A function that will encapsulate the functions declared in this script and execute them in order to prepare features"""

    processed_data = data.copy()

    if "bathrooms" in config["columns_for_data_prep"]:
        processed_data = get_bathroom_nums(data, config["bathroom_text_column"])

    columns_to_keep = config["columns_for_data_prep"].copy()
    columns_to_keep.append(config["target_variable"])

    processed_data = processed_data[columns_to_keep]
    processed_data = processed_data.dropna(axis=0)

    processed_data = neighbourhood_transformations(data=processed_data)

    processed_data = room_type_transformations(data=processed_data)

    processed_data = binary_columns_from_text(
        data=processed_data,
        column_to_check=config["text_column_for_binary_generation"],
        columns_names=config["binary_column_names_to_generate_by_text"],
        columns_text_to_check=config["text_to_check"],
    )

    return processed_data
