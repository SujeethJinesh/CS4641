import pandas as pd
from sklearn import preprocessing


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


def _normalizer(df, attribute_list):
    """
        Takes in list of attributes
        Ex:
            ["age"]
        :return: New, replaceable dataframe with replaced values
    """
    # for normalizing
    min_max_scaler = preprocessing.MinMaxScaler()

    for attribute in attribute_list:
        # df_scaled = min_max_scaler.fit_transform(df[attribute].astype(int).values.reshape(-1, 1))
        df_scaled = preprocessing.normalize(df[attribute].astype(float).values.reshape(-1, 1))
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


def getCleanData(file_path, attributes, binary_one_hot_map=None, normalize_list=None, multi_one_hot_list=None,
                 cols_to_drop=None, row_num_to_drop=None):
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

    # replace binary variables with 1/0
    if not (binary_one_hot_map is None):
        data = _binaryOneHot(data, binary_one_hot_map)

    # normalize values to prevent them from having too much of a weight
    if not (normalize_list is None):
        data = _normalizer(data, normalize_list)

    # one-hot encode multiple categorical values
    if not (multi_one_hot_list is None):
        data = _multiOneHot(data, multi_one_hot_list)

    return data

