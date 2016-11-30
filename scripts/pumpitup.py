'''
Helper functions to use for DrivenData's Pump It Up challenge.
'''
import datetime as dt
import pandas as pd
import numpy as np
import re

from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold


def unique_count(df):
    '''
    Prints the name of each feature column and the number of unique elements that 
    it contains.

    Returns:
    df_uniques - dataframe with two columns for the feature names and the number
                 of unique elements corresponding to the feature.
    '''
    d = {'Feature':[], 'Uniques': []}
    for column in df.columns:
        d['Feature'].append(column)
        d['Uniques'].append(len(df[column].unique()))

    df_uniques = pd.DataFrame(d)
    return df_uniques


def remove_features(df, features=None):
    '''
    Removes id, num_private and recorded_by features from data frame as standard.
    Removes any other features specified in features list.

    Returns:
    df - original data frame with specified feature columns removed
    '''
    df.drop(['id', 'num_private', 'recorded_by'], axis=1, inplace=True)
    if features:
        df.drop(features, axis=1, inplace=True)

    return df


def percent_missing(df):
    '''
    Prints a list showing the percentage of missing values in each feature 
    column plus their types (0, 'NaN' or 'none')

    Returns:
    percent_missing - list of the percent of missing values
    '''
    percent_missing = []

    total = len(df)

    for column in df.columns:
        series = df[column]

        if series.dtype == 'object':
            if series.isnull().any():
                percentage = (sum(series.isnull()) / total)  * 100
                print(column, '{:.6f}'.format(percentage), 'NaN')

            elif series.str.contains('none').any():
                num = len(series[series == 'none'])
                percentage = (num / total) * 100
                print(column, '{:.6f}'.format(percentage), 'none')

        elif series.dtype == 'float64':
            num = sum(abs(series) < 1e-6)
            percentage = num / total * 100
            print(column, '{:.6f}'.format(percentage), '0')

        elif series.dtype == 'int64':
            num = sum(series == 0)
            percentage = num / total * 100
            print(column, '{:.6f}'.format(percentage), '0')

    return percent_missing


def convert_dates(df, column='date_recorded'):
    '''
    Converts date object feature into useful time values.

    Returns:
    df - original data frame with 'date_recorded' column removed and replaced
         by days since epoch and year month.
    '''
    epoch_day = dt.datetime.utcfromtimestamp(0)
    
    dates = [(dt.datetime.strptime(date, '%Y-%m-%d')) for date in df[column]]

    years = [date.year for date in dates]
    df['year_recorded'] = years

    year_months = [date.month for date in dates]
    df['month_recorded'] = year_months

    #epoch_days = [(date -  epoch_day).days for date in dates]
    #df['epoch_day'] = epoch_days

    df['operation_years'] = df['year_recorded'] - df['construction_year']

    df.drop(column, axis=1, inplace=True)

    return df


def classification_rate(y_actual, y_predicted):
    '''
    Calculates the fraction of times where y_predicted is equal to y_actual.

    Returns:
    rate - the classification rate
    '''
    rate = np.sum(y_actual == y_predicted) / len(y_actual)
    return rate


def nan2str(df):
    '''
    Convert missing values of type np.nan to string 'nan'
    '''
    df[(pd.isnull(df))] = 'none'


def label_encode(series, label_encoder):
    '''
    Encodes specified columns using label encoding.

    Returns:
    encoded - series of categories encoded as integers.
    '''
    encoded = label_encoder.fit_transform(series)
    return encoded


def run_forest(df_X, df_y, n_folds=5, n_estimators=1000):
    '''
    Fold data into training and cv sets then train random forest classifier.

    Returns:
    predictions - the predictions obtained from each fold of the input data
    clf - the trained classifier
    '''
    kf = KFold(df_X.shape[0], n_folds=n_folds, random_state=123456)

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=2)
    
    predictions = []

    for train, cv in kf:
        X = df_X.iloc[train, :]
        y = df_y['status_group'].iloc[train]

        clf.fit(X, y)

        fold_predictions = clf.predict(df_X.iloc[cv, :])
        predictions = np.append(predictions, fold_predictions)

    return predictions, clf


def fill_missing_simple(df, columns):
    '''
    Replaces missing values with modal value for string features and median 
    value for numeric features.

    Returns:
    df - original dataframe with missing values replaced
    '''
    for column in columns:
        series = df[column]
        data_type = series.dtype

        if data_type == 'object':
            # if the missing value is the modal value then choose the second 
            # most common
            if series.mode()[0] == 'none':
                series_present = series[series != 'none']
                most_common = series_temp.mode()[0]
            # otherwise find the most common value
            else:
                most_common = series.mode()[0]

            # replace all missing values in column with the modal value
            df[column][series == 'none'] = most_common

        elif (data_type == 'float64') or (data_type == 'int64'):
            # find locations of very small values
            missing = abs(0 - abs(series)) < 10e-6
            # create series of only rows with non-missing values
            series_present = series[missing ==  False]
            med = series_present.median()
            # convert to int if necessary
            if data_type == 'int64':
                med = int(med)

            # replace all missing values in column with the median value
            df[column][missing] = med

    return df


def flag_missing_df(df):
    '''
    Find the positions of all missing values in a data frame.

    Returns:
    df_flag_missing - a boolean data frame where True denotes a missing value.
    missing_headers - a list of the header names of columns where missing 
                      values are present.
    '''
    df_flag_missing = df.copy()

    missing_headers = []

    for column in df.columns:
        series = df[column]
        data_type = series.dtype

    if data_type == 'object':
        flags = series.str.contains('none')
        df_flag_missing[column] = flags

        if flags.any():
            missing_headers.append(column)

    elif data_type == 'float64':
        flags = abs(0 - abs(series)) < 10e-6
        df_flag_missing[column] = flags
        
        if flags.any():
            missing_headers.append(column)

    elif data_type == 'int64':
        flags = series == 0
        df_flag_missing = flags

        if flags.any():
            missing_headers.append(column)

    return df_flag_missing, missing_headers

def flag_missing_s(series):
    '''
    Find the positions of all missing values in a data frame.

    Returns:
    df_flag_missing - a boolean data frame where True denotes a missing value.
    missing_headers - a list of the header names of columns where missing 
                      values are present.
    '''
    data_type = series.dtype

    if data_type == 'object':
        flags = series.str.contains('none')
        series_flag_missing[column] = flags

    elif data_type == 'float64':
        flags = abs(0 - abs(series)) < 10e-6
        series_flag_missing[column] = flags

    elif data_type == 'int64':
        flags = series == 0
        series_flag_missing = flags

    return series_flag_missing

def binary_count(series):
    '''
    Calculates length of the number of unique categories, expressed as a binary.

    Returns:
    binary_digits - integer representing the number of binary digits needed to 
                    display the number of unique values in a feature.
    '''
    binary_digits = len('{0:b}'.format(len(s.unique())))

    return binary_digits


def binary_encode(series, le):
    '''
    Encodes categorical data as binary by label encoding them and then 
    converting to binary.

    Returns:
    binary_encoded - a series of the category values as binary numbers in 
                     string form.
    binary_columns - a data frame with each column expressing the corresponding 
                     digit of the binary encoded values.

    '''

    series =  series.apply(lambda x: str(x))

    binary_digits = binary_count(series)

    label_encoded = le.fit_transform(series)

    binary_encoded = ['{0:b}'.format(label).zfill(binary_digits) for label \
                         in label_encoded]
    binary_columns = [list(binary) for binary in binary_encoded]

    binary_encoded = pd.Series(binary_encoded)

    binary_headers = [column + '_' + str(i) for i in range(0, binary_digits)]
    binary_columns = pd.DataFrame(binary_columns, columns=binary_headers)

    return binary_encoded, binary_columns, binary_headers

def fill_missing_knn(series, series_missing_flags, df_encoded, k=5):
    '''
    Find the missing values in a series by finding k nearest neighbours and 
    using their information to fill in the missing data.

    Returns:
    series - original series with missing values filled in by knn algorithm
    '''
    data_type = series.dtype

    series_exist_flags = series_missing_flags == False
    series_missing = series[series_missing_flags]
    series_exist = series[series_exist_flags]

    df_missing = df_encoded[series_missing_flags]
    df_exist = df_encoded[series_exist_flags]

    neigh = KNeighborsClassifier(n_neighbors=k)

    series_exist = series_exist.values.reshape(-1)

    label, indices = neigh.kneigbors(df_exist, n_neighbors=k)

    neigh.fit(df_exist, series_exist)

    if dtype == 'float64':
        missing_means = [np.mean(series_exist[i]) for i in indices]
        series[series_missing_flags] = missing_means

    elif dtype == 'int64':
        missing_means = [int(np.mean(series_exist[i])) for i in indices]
        series[series_missing_flags] = missing_means

    elif dtype == 'object':
        missing_mode = [stats.mode(series_exist[i]) for i in indices]
        #missing_mode = [list(list(x.mode)[0]) for x in missing_mode]
        series[series_missing_flags] = missing_mode

    return series        





