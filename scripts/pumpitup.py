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
from sklearn import svm
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


def run_forest(df_X, df_y, n_folds=5, n_estimators=1000, n_jobs=3, max_features=10,\
               max_depth=None, min_samples_split=2):
    '''
    Fold data into training and cv sets then train random forest classifier.

    Returns:
    predictions - the predictions obtained from each fold of the input data
    clf - the trained classifier
    '''
    kf = KFold(df_X.shape[0], n_folds=n_folds, random_state=123456)

    clf = RandomForestClassifier(n_estimators=n_estimators,\
        n_jobs=n_jobs, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split)
    
    predictions = []

    for train, cv in kf:
        X = df_X.iloc[train, :]
        y = df_y['status_group'].iloc[train]

        clf.fit(X, y)

        fold_predictions = clf.predict(df_X.iloc[cv, :])
        predictions = np.append(predictions, fold_predictions)

    return predictions, clf

def run_svm(df_X, df_y, n_folds=5):
    '''
    Fold data into training and cv sets then train random forest classifier.

    Returns:
    predictions - the predictions obtained from each fold of the input data
    clf - the trained classifier
    '''
    kf = KFold(df_X.shape[0], n_folds=n_folds)

    clf = svm.SVC()
    
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

    elif data_type == 'float64':
        flags = abs(0 - abs(series)) < 10e-6

    elif data_type == 'int64':
        flags = series == 0

    return flags

def missing2nan(series):
    flags = flag_missing_s(series)
    series[flags] = np.nan
    return series

def binary_count(series):
    '''
    Calculates length of the number of unique categories, expressed as a binary.

    Returns:
    binary_digits - integer representing the number of binary digits needed to 
                    display the number of unique values in a feature.
    '''
    #binary_digits = len('{0:b}'.format(len(series.unique())))
    binary_digits = len('{0:b}'.format(len(series.unique())))

    return binary_digits


def binary_encode(series, le, binary_lengths=None):
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

    if binary_lengths:
        binary_digits = binary_lengths
    else:
        binary_digits = binary_count(series)

    label_encoded = le.fit_transform(series)

    binary_encoded = ['{0:b}'.format(label).zfill(binary_digits) for label \
                         in label_encoded]
    binary_columns = [list(binary) for binary in binary_encoded]

    binary_encoded = pd.Series(binary_encoded)

    binary_headers = [series.name + '_' + str(i) for i in range(0, binary_digits)]
    binary_columns = pd.DataFrame(binary_columns, columns=binary_headers)

    return binary_encoded, binary_columns, binary_headers

def fill_missing_knn(series, df_encoded, k=5):
    '''
    Find the missing values in a series by finding k nearest neighbours and 
    using their information to fill in the missing data.

    Returns:
    series - original series with missing values filled in by knn algorithm
    '''
    series_missing_flags = flag_missing_s(series)
    data_type = series.dtype

    series_exist_flags = series_missing_flags == False
    series_missing = series[series_missing_flags]
    series_exist = series[series_exist_flags]

    df_missing = df_encoded[series_missing_flags]
    df_exist = df_encoded[series_exist_flags]

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(df_exist, series_exist)

    series_exist = series_exist.values.reshape(-1)

    label, indices = neigh.kneighbors(df_missing, n_neighbors=k)

    #neigh.fit(df_exist, series_exist)

    if data_type == 'float64':
        missing_means = [np.mean(series_exist[i]) for i in indices]
        series[series_missing_flags] = missing_means

    elif data_type == 'int64':
        missing_means = [int(np.mean(series_exist[i])) for i in indices]
        series[series_missing_flags] = missing_means

    elif data_type == 'object':
        missing_mode = [stats.mode(series_exist[i]) for i in indices]
        #missing_mode = [list(list(x.mode)[0]) for x in missing_mode]
        series[series_missing_flags] = missing_mode

    return series        

def cleanitup(df):

    min_cat_size = 0.0084 * len(df)
    
    modifiers = {}
    
    df.drop(['id', 'recorded_by', 'num_private'], axis=1, inplace=True)
    
    # EXTRACTION TYPES
    # rename some categories in the finest featureset
    # drop mid level information in feature extraction_type_group
    df['extraction_type'].replace('other - swn 81', 'swn 81', inplace=True)
    df['extraction_type'].replace('other - mkulima/shinyanga', 'other handpump', inplace=True)
    df['extraction_type'].replace('other - play pump', 'other handpump', inplace=True)
    df['extraction_type'].replace('cemo', 'other motorpump', inplace=True)
    df['extraction_type'].replace('climax', 'other motorpump', inplace=True)
    df.drop('extraction_type_group', axis=1, inplace=True)
    
    # MANAGEMENT
    # do nothing
    
    # SCHEME
    # drop scheme_name as too many categories and missing values
    # also drop scheme_management as basically the same as management
    df.drop('scheme_name', axis=1, inplace=True)
    df.drop('scheme_management', axis=1, inplace=True)
    
    # PAYMENT
    # two identical features - drop payment
    df.drop('payment', axis=1, inplace=True)
    
    # WATER QUALITY
    # water_quality contains slightly more information than quality_group
    df.drop('quality_group', axis=1, inplace=True)
    
    # QUANTITY
    # two identical groups - drop quantity_group
    df.drop('quantity_group', axis=1, inplace=True)
    
    #SOURCE
    # drop mid level source_type
    df.drop('source_type', axis=1, inplace=True)
    df['source'].replace('other', 'unknown', inplace=True)
    
    # WATERPOINT
    # drop waterpoint_type_group as less precise than waterpoint_type
    df.drop('waterpoint_type_group', axis=1, inplace=True)
    
    # GEOGRAPHICAL
    # drop mid level geographic information
    # keep district_code as this comes in useful later!
    df.drop(['region_code', 'subvillage', 'ward'], axis=1, inplace=True)
    
    lga = df['lga'].copy()
    lga[lga.str.contains('Rural')] = 'rural'
    lga[lga.str.contains('Urban')] = 'urban'
    other_flag = lga.str.contains('rural') | lga.str.contains('urban')
    other_flag = other_flag == False
    lga[other_flag] = 'other'
    df['lga'] = lga
    
    # TIME
    # convert date_recorded to days since epoch and year month
    df = convert_dates(df)
    
    # WATERPOINT NAME
    df['wpt_name'] = df['wpt_name'].str.lower()
    df['wpt_name'][df['wpt_name'].str.contains('school')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('shule')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('sekondari')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('secondary')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('sekondari')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('msingi')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('primary')] = 'school'

    df['wpt_name'][df['wpt_name'].str.contains('clinic')] = 'health'
    df['wpt_name'][df['wpt_name'].str.contains('zahanati')] = 'health'
    df['wpt_name'][df['wpt_name'].str.contains('health')] = 'health'
    df['wpt_name'][df['wpt_name'].str.contains('hospital')] = 'health'

    df['wpt_name'][df['wpt_name'].str.contains('ccm')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('office')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('kijiji')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('ofis')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('idara')] = 'official'

    df['wpt_name'][df['wpt_name'].str.contains('farm')] = 'farm'
    df['wpt_name'][df['wpt_name'].str.contains('maziwa')] = 'farm'

    df['wpt_name'][df['wpt_name'].str.contains('pump house')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('pump')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('bombani')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('maji')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('water')] = 'water'

    df['wpt_name'][df['wpt_name'].str.contains('kanisani')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('kanisa')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('church')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('luther')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('anglican')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('pentecost')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('msikitini')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('msikiti')] = 'religious'

    df['wpt_name'][df['wpt_name'].str.contains('center')] = 'center'
    df['wpt_name'][df['wpt_name'].str.contains('market')] = 'center'
    df['wpt_name'][df['wpt_name'].str.contains('sokoni')] = 'center'
    df['wpt_name'][df['wpt_name'].str.contains('madukani')] = 'center'

    df['wpt_name'][df['wpt_name'].str.contains('kwa')] = 'name'

    #finally change any values with less than 500 records to 'other' as well as the 'none' values
    value_counts = df['wpt_name'].value_counts()
    to_remove = value_counts[value_counts <= min_cat_size].index
    df['wpt_name'].replace(to_remove, 'other', inplace=True)

    df['wpt_name'][df['wpt_name'].str.contains('none')] = 'other'
    modifiers['wpt_name'] = df['wpt_name'].value_counts().index.tolist()

    
    # INSTALLER AND FUNDER
    value_counts = df['funder'].value_counts()
    to_remove = value_counts[value_counts <= min_cat_size].index
    df['funder'].replace(to_remove, 'other', inplace=True)
    value_counts = df['installer'].value_counts()
    to_remove = value_counts[value_counts <= min_cat_size].index
    df['installer'].replace(to_remove, 'other', inplace=True)

    modifiers['funder'] = df['funder'].value_counts().index.tolist()
    modifiers['installer'] = df['installer'].value_counts().index.tolist()
    
    ### MISSING DATA ###
    df.drop('amount_tsh', axis=1, inplace=True)
    
    # LONG - LAT
    # Use region and district_codes to fill in missing longitude and latitude data
    mask1 = flag_missing_s(df['longitude'])
    df['longitude'][mask1] = np.nan
    df.loc[mask1, 'longitude'] = df.groupby(['region', 'district_code']).transform('mean')
    mask2 = df['longitude'].isnull()
    df.loc[mask2, 'longitude'] = df.groupby(['region']).transform('mean')
    
    mask3 = flag_missing_s(df['latitude'])
    df['latitude'][mask3] = np.nan
    df.loc[mask3, 'latitude'] = df.groupby(['region', 'district_code']).transform('mean')
    mask4 = df['latitude'].isnull()
    df.loc[mask4, 'latitude'] = df.groupby(['region']).transform('mean')
    
    # GPS HEIGHT
    # Use KNN imputation of the longitude and latitude variables
    df_encoded = df[['latitude', 'longitude']]
    series = df.gps_height
    height_filled = fill_missing_knn(series, df_encoded, k=5)
    df['gps_height'] = height_filled
    
    # POPULATION
    # use grouping of geographical area, region and district code to impute missing population data
    mask1 = flag_missing_s(df['population'])
    df['population'][mask1] = np.nan
    df.loc[mask1, 'population'] = df.groupby(['lga', 'region','district_code']).transform('mean')
    mask2 = df['population'].isnull()
    df.loc[mask2, 'population'] = df.groupby(['lga', 'region']).transform('mean')
    mask3 = df['population'].isnull()
    df.loc[mask3, 'population'] = df.groupby('lga').transform('mean')
    
    df['population'] = df['population'].astype(int)
    
    # TIME
    df['year_recorded'][df['year_recorded'] < df['operation_years']] = df['year_recorded'].median()
    
    mask1 = flag_missing_s(df['construction_year'])
    df['construction_year'][mask1] = np.nan
    df.loc[mask1, 'construction_year'] = df.groupby(['extraction_type', 'installer']).transform('median')
    mask2 = df['construction_year'].isnull()
    df.loc[mask2, 'construction_year'] = df.groupby(['extraction_type']).transform('median')
    mask3 = df['construction_year'].isnull()
    df['construction_year'][mask3] = df['construction_year'].median()
    
    df['construction_year'] = df['construction_year'].astype(int)
    
    df['operation_years'] = (df['year_recorded'] - df['construction_year']).astype(int)
    df['operation_years'][df['operation_years'] < 0] = df['operation_years'].median()
    
    # INSTALLER AND FUNDER
    mask = df['installer'].isnull()
    df['installer'][mask] = 'other'
    df.loc[mask, 'installer'] = df.groupby(['region', 'district_code'])['installer']\
        .transform(lambda x: x.value_counts().index[0])
    mask = df['funder'].isnull()
    df['funder'][mask] = 'other'
    df.loc[mask, 'funder'] = df.groupby(['region', 'district_code'])['funder']\
        .transform(lambda x: x.value_counts().index[0])
        
    # PERMIT AND PUBLIC MEETING
    mask = df['permit'].isnull()
    df['permit'][mask] = True
    df.loc[mask, 'permit'] = df.groupby(['region', 'district_code'])['permit'].transform(lambda x: x.value_counts().index[0])

    mask = df['public_meeting'].isnull()
    df['public_meeting'][mask] = True
    df.loc[mask, 'public_meeting'] = df.groupby(['region', 'district_code'])['public_meeting'].transform(lambda x: x.value_counts().index[0])
    
    df.drop('district_code', axis=1, inplace=True)
    
    return df, modifiers


def cleantestup(df, modifiers):
    
    # set minimum size for a category (0.0084 is approx 500 samples in training data)
    min_cat_size = 0.0084 * len(df)
    
    # drop immediately unhelpful data
    df.drop(['id', 'recorded_by', 'num_private'], axis=1, inplace=True)
    
    # EXTRACTION TYPES
    # rename some categories in the finest featureset
    # drop mid level information in feature extraction_type_group
    df['extraction_type'].replace('other - swn 81', 'swn 81', inplace=True)
    df['extraction_type'].replace('other - mkulima/shinyanga', 'other handpump', inplace=True)
    df['extraction_type'].replace('other - play pump', 'other handpump', inplace=True)
    df['extraction_type'].replace('cemo', 'other motorpump', inplace=True)
    df['extraction_type'].replace('climax', 'other motorpump', inplace=True)
    df.drop('extraction_type_group', axis=1, inplace=True)
    
    # MANAGEMENT
    # do nothing
    
    # SCHEME
    # drop scheme_name as too many categories and missing values
    # also drop scheme_management as basically the same as management
    df.drop('scheme_name', axis=1, inplace=True)
    df.drop('scheme_management', axis=1, inplace=True)
    
    # PAYMENT
    # two identical features - drop payment
    df.drop('payment', axis=1, inplace=True)
    
    # WATER QUALITY
    # water_quality contains slightly more information than quality_group
    df.drop('quality_group', axis=1, inplace=True)
    
    # QUANTITY
    # two identical groups - drop quantity_group
    df.drop('quantity_group', axis=1, inplace=True)
    
    #SOURCE
    # drop mid level source_type
    df.drop('source_type', axis=1, inplace=True)
    df['source'].replace('other', 'unknown', inplace=True)
    
    # WATERPOINT
    # drop waterpoint_type_group as less precise than waterpoint_type
    df.drop('waterpoint_type_group', axis=1, inplace=True)
    
    # GEOGRAPHICAL
    # drop mid level geographic information
    # keep district_code as this comes in useful later!
    df.drop(['region_code', 'subvillage', 'ward'], axis=1, inplace=True)
    
    lga = df['lga'].copy()
    lga[lga.str.contains('Rural')] = 'rural'
    lga[lga.str.contains('Urban')] = 'urban'
    other_flag = lga.str.contains('rural') | lga.str.contains('urban')
    other_flag = other_flag == False
    lga[other_flag] = 'other'
    df['lga'] = lga
    
    # TIME
    # convert date_recorded to days since epoch and year month
    df = convert_dates(df)

    # WATERPOINT NAME
    # group waterpoint names that are similar into categories
    df['wpt_name'] = df['wpt_name'].str.lower()
    df['wpt_name'][df['wpt_name'].str.contains('school')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('shule')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('sekondari')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('secondary')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('sekondari')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('msingi')] = 'school'
    df['wpt_name'][df['wpt_name'].str.contains('primary')] = 'school'

    df['wpt_name'][df['wpt_name'].str.contains('clinic')] = 'health'
    df['wpt_name'][df['wpt_name'].str.contains('zahanati')] = 'health'
    df['wpt_name'][df['wpt_name'].str.contains('health')] = 'health'
    df['wpt_name'][df['wpt_name'].str.contains('hospital')] = 'health'

    df['wpt_name'][df['wpt_name'].str.contains('ccm')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('office')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('kijiji')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('ofis')] = 'official'
    df['wpt_name'][df['wpt_name'].str.contains('idara')] = 'official'

    df['wpt_name'][df['wpt_name'].str.contains('farm')] = 'farm'
    df['wpt_name'][df['wpt_name'].str.contains('maziwa')] = 'farm'

    df['wpt_name'][df['wpt_name'].str.contains('pump house')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('pump')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('bombani')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('maji')] = 'water'
    df['wpt_name'][df['wpt_name'].str.contains('water')] = 'water'

    df['wpt_name'][df['wpt_name'].str.contains('kanisani')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('kanisa')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('church')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('luther')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('anglican')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('pentecost')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('msikitini')] = 'religious'
    df['wpt_name'][df['wpt_name'].str.contains('msikiti')] = 'religious'

    df['wpt_name'][df['wpt_name'].str.contains('center')] = 'center'
    df['wpt_name'][df['wpt_name'].str.contains('market')] = 'center'
    df['wpt_name'][df['wpt_name'].str.contains('sokoni')] = 'center'
    df['wpt_name'][df['wpt_name'].str.contains('madukani')] = 'center'

    df['wpt_name'][df['wpt_name'].str.contains('kwa')] = 'name'

    # change any new or different categories to 'other'
    mask = df['wpt_name'].isin(modifiers['wpt_name'])
    mask = [not val for val in mask]
    df['wpt_name'][mask] = 'other'

    # INSTALLER AND FUNDER
    # change any funder and installer categories with counts less than threshold to 'other'
    value_counts = df['funder'].value_counts()
    to_remove = value_counts[value_counts <= min_cat_size].index
    df['funder'].replace(to_remove, 'other', inplace=True)
    value_counts = df['installer'].value_counts()
    to_remove = value_counts[value_counts <= min_cat_size].index
    df['installer'].replace(to_remove, 'other', inplace=True)

    # change any new or different categories to 'other'
    mask = df['funder'].isin(modifiers['funder'])
    mask = [not val for val in mask]
    df['funder'][mask] = 'other'
    # change any new or different categories to 'other'
    mask = df['installer'].isin(modifiers['installer'])
    mask = [not val for val in mask]
    df['installer'][mask] = 'installer'

        ### MISSING DATA ###
    df.drop('amount_tsh', axis=1, inplace=True)
    
    # LONG - LAT
    # Use region and district_codes to fill in missing longitude and latitude data
    mask1 = flag_missing_s(df['longitude'])
    df['longitude'][mask1] = np.nan
    df.loc[mask1, 'longitude'] = df.groupby(['region', 'district_code']).transform('mean')
    mask2 = df['longitude'].isnull()
    df.loc[mask2, 'longitude'] = df.groupby(['region']).transform('mean')
    
    mask3 = flag_missing_s(df['latitude'])
    df['latitude'][mask3] = np.nan
    df.loc[mask3, 'latitude'] = df.groupby(['region', 'district_code']).transform('mean')
    mask4 = df['latitude'].isnull()
    df.loc[mask4, 'latitude'] = df.groupby(['region']).transform('mean')
    
    # GPS HEIGHT
    # Use KNN imputation of the longitude and latitude variables
    df_encoded = df[['latitude', 'longitude']]
    series = df.gps_height
    height_filled = fill_missing_knn(series, df_encoded, k=5)
    df['gps_height'] = height_filled
    
    # POPULATION
    # use grouping of geographical area, region and district code to impute missing population data
    mask1 = flag_missing_s(df['population'])
    df['population'][mask1] = np.nan
    df.loc[mask1, 'population'] = df.groupby(['lga', 'region','district_code']).transform('mean')
    mask2 = df['population'].isnull()
    df.loc[mask2, 'population'] = df.groupby(['lga', 'region']).transform('mean')
    mask3 = df['population'].isnull()
    df.loc[mask3, 'population'] = df.groupby('lga').transform('mean')
    
    df['population'] = df['population'].astype(int)
    
    # TIME
    df['year_recorded'][df['year_recorded'] < df['operation_years']] = df['year_recorded'].median()
    
    mask1 = flag_missing_s(df['construction_year'])
    df['construction_year'][mask1] = np.nan
    df.loc[mask1, 'construction_year'] = df.groupby(['extraction_type', 'installer']).transform('median')
    mask2 = df['construction_year'].isnull()
    df.loc[mask2, 'construction_year'] = df.groupby(['extraction_type']).transform('median')
    mask3 = df['construction_year'].isnull()
    df['construction_year'][mask3] = df['construction_year'].median()
    
    df['construction_year'] = df['construction_year'].astype(int)
    
    df['operation_years'] = (df['year_recorded'] - df['construction_year']).astype(int)
    df['operation_years'][df['operation_years'] < 0] = df['operation_years'].median()
    
    # INSTALLER AND FUNDER
    mask = df['installer'].isnull()
    df['installer'][mask] = 'other'
    df.loc[mask, 'installer'] = df.groupby(['region', 'district_code'])['installer']\
        .transform(lambda x: x.value_counts().index[0])
    mask = df['funder'].isnull()
    df['funder'][mask] = 'other'
    df.loc[mask, 'funder'] = df.groupby(['region', 'district_code'])['funder']\
        .transform(lambda x: x.value_counts().index[0])
        
    # PERMIT AND PUBLIC MEETING
    mask = df['permit'].isnull()
    df['permit'][mask] = True
    df.loc[mask, 'permit'] = df.groupby(['region', 'district_code'])['permit'].transform(lambda x: x.value_counts().index[0])

    mask = df['public_meeting'].isnull()
    df['public_meeting'][mask] = True
    df.loc[mask, 'public_meeting'] = df.groupby(['region', 'district_code'])['public_meeting'].transform(lambda x: x.value_counts().index[0])
    
    df.drop('district_code', axis=1, inplace=True)

    return df

