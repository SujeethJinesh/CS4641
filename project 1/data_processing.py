import pandas as pd
from sklearn import preprocessing
import numpy as np


def _binaryOneHot(df, attribute_map):
    """
        Takes in map with keys of attributes, and value of a list of the 0 and 1 values
        Ex:
            {
                "school": ["GP", "MS"]
                "sex": ["F","M"]
            }
        :return: New, replaceable dataframe with replaced values
    """
    for key in attribute_map:
        df[key] = df[key].map({attribute_map[key][0]: 0, attribute_map[key][1]: 1})

    return df


def _normalizer(df, attribute_list, normalizer_type=None):
    """
        Takes in list of attributes
        Ex:
            ["age"]
        :return: New, replaceable dataframe with replaced values
    """
    # for normalizing
    min_max_scaler = preprocessing.MinMaxScaler()

    # import ipdb; ipdb.set_trace()

    for attribute in attribute_list:
        if not normalizer_type:
            df_scaled = preprocessing.normalize(df[attribute].astype(float).values.reshape(-1, 1))
        elif normalizer_type == 'min_max':
            df_scaled = min_max_scaler.fit_transform(df[attribute].astype(int).values.reshape(-1, 1))
        df[attribute] = df_scaled

    return df


def _multiOneHot(df, attribute_list):
    """
        Takes in list of attributes
        Ex:
            ["age"]
        :return: New, replaceable dataframe with replaced values
    """
    for attribute in attribute_list:
        one_hot = pd.get_dummies(df[attribute], prefix=attribute)
        df = pd.concat([df, one_hot], axis=1)
        df.drop([attribute], axis=1, inplace=True)

    return df


def missing_data_fixer(df, missing_data_marker, to_impute):
    df = df.replace(missing_data_marker, np.nan)
    # df.applymap(lambda x: np.nan if x == missing_data_marker else x)
    df.drop(['level_0', 'index'], inplace=True, axis=1, errors='ignore')
    if to_impute:
        df.fillna(df.mean())
    else:
        df.dropna(axis=0, inplace=True) #drop rows with missing data
    # import ipdb; ipdb.set_trace()
    return df


def getCleanData(file_path, attributes, binary_one_hot_map=None, normalize_list=None, multi_one_hot_list=None,
                 cols_to_drop=None, row_num_to_drop=None, missing_data_marker=None, to_impute=None, normalizer_type=None):
    """
    :return: Cleaned data
    """
    # read in data
    data = pd.read_csv(file_path, names=attributes)

    # drop col_to_drop column
    if not (cols_to_drop is None):
        for col in cols_to_drop:
            data = data.drop(col, axis=1)

    # dropping rows we don't need
    if not (row_num_to_drop is None):
        data.drop(data.index[row_num_to_drop], inplace=True)

    # missing data marker
    if not (missing_data_marker is None):
        data = missing_data_fixer(data, missing_data_marker, to_impute)

    # replace binary variables with 1/0
    if not (binary_one_hot_map is None):
        data = _binaryOneHot(data, binary_one_hot_map)

    # normalize values to prevent them from having too much of a weight
    if not (normalize_list is None):
        data = _normalizer(data, normalize_list, normalizer_type=normalizer_type)

    # one-hot encode multiple categorical values
    if not (multi_one_hot_list is None):
        data = _multiOneHot(data, multi_one_hot_list)

    return data

