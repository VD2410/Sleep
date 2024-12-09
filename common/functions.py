import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import re
import os
import warnings
import tensorflow as tf
import joblib
from common.logger import LoggerConfig
warnings.filterwarnings('ignore')

def read_text_file(file_path):
    """This function read's the input path of text file and returns the data inside the file

    Args:
        file_path (str): File path

    Returns:
        line (str): data stored as string
    """
    try:
        with open(file_path) as f:
            line = f.read()
    except Exception as e:
            logger = LoggerConfig()
            logger.configure_logger()
            logger.log_error(str(e))
    else:
        return line

def read_csv_file(file_path, index_columns=None):
    """This function read's the input path of csv file and returns the data inside the file

    Args:
        file_path (str): File path
        index_columns (list): list of column index that will be used as index

    Returns:
        df (dataframe): dataframe (of the csv file)
    """
    try:
        df = pd.read_csv(file_path, index_col=index_columns)
    except Exception as e:
            logger = LoggerConfig()
            logger.configure_logger()
            logger.log_error(str(e))
    else:
        return df
    
    

def save_csv_file(df, file_path):
    """This function read's the filepath and saves the dataframe to that folder

    Args:
        df (dataframe): dataframe to be saved into a csv file
        file_path (str): file path
    """
    df.to_csv(file_path)
    

def extract_user_data_from_combined_data(df, user_id):
    """This function will extract user data from the dataframe

    Args:
        df (dataframe): Dataframe from where the user information will be extrated
        user_id (string): user_id who's information will be extracted

    Returns:
        user_df (dataframe): Dataframe with user information
    """
    user_df = df[df['user_id'] == user_id]
    user_df.reset_index(inplace= True, drop= True)
    return user_df
    

    
def extract_user_id(input):
    """This function will extract the USER ID from the provided input using regex

    Args:
        input (list): list which contains the user id

    Returns:
        user_ids (list): list of user_ids
    """
    user_ids = input.split(',')
    user_ids = [user_id.strip() for user_id in user_ids]
    user_ids = [re.findall(r"^([^.\s]*)", user_id)[0] for user_id in user_ids if len(re.findall(r"^([^.\s]*)", user_id)[0]) > 0]
    return user_ids

def merge_data(*dataframes):
    """This function will merge the dataframes vertically

    Args:
        dataframes (dataframe): dataframes to merge

    Returns:
        merged_df (dataframe): merged dataframe
    """
    merged_df = pd.DataFrame()
    for df in dataframes:
        merged_df = pd.concat([merged_df, df], axis=0)
    merged_df.reset_index(inplace=True, drop=True)
    return merged_df

def remove_duplicate_values(df, columns=None):
    """This function will remove the duplicates from the dataframe

    Args:
        df (dataframe): Dataframe from which duplicates needs to be removed
        columns (list): Will check duplicates in the columns provided. Defaults to None.

    Returns:
        filtered_df (dataframe): Dataframe with no duplicates
    """
    filtered_df = df.drop_duplicates(subset=columns)
    filtered_df.reset_index(inplace= True, drop= True)
    return filtered_df

def check_dir_exists_else_create(dir_path):
    """This function will create directory if it does not exist

    Args:
        dir_path (str): path to the directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_dir_exists(dir_path):
    """This function will check if the directory exists

    Args:
        dir_path (str): path to the directory

    Returns:
        path_exists (bool): return if the path exists or not 
    """
    return os.path.exists(dir_path)
            


def attribute_datatype_change_to_datetime(df, column, errors='raise'):
    """This function will change the datatype of an attribute to datetime

    Args:
        df (dataframe): dataframe which contains the attribute
        column (str): attribute name
        error (str): type of error

    Returns:
        df[column] (series): series with attribute datatype as datetime
    """
    df[column] = pd.to_datetime(df[column], errors=errors)
    return df[column]

def adjust_shift_start_time_for_off_days(df):
    """This function will adjust shift start time for off days to 1 minute difference between start time and end time

    Args:
        df (dataframe): dataframe with off days

    Returns:
        df (dataframe): dataframe with changed start time 
    """
    df.loc[df['off_day'] == 1, 'start_time'] = df.loc[df['off_day'] == 1, 'end_time'] - timedelta(minutes=1)
    return df

def drop_null_values(df, columns=None):
    """This function will remove the duplicates from the dataframe

    Args:
        df (dataframe): Dataframe from which duplicates needs to be removed
        columns (list): Will check null values in the columns provided. Defaults to None.

    Returns:
        filtered_df (dataframe): Dataframe from which null values were removed
    """
    filtered_df = df.dropna(subset= columns)
    filtered_df.reset_index(inplace= True, drop= True)
    return filtered_df

def sort_values(df, columns, order=True):
    """This function will sort the values based on the order provided

    Args:
        df (dataframe): Dataframe which needs to be sorted
        columns (list): list of features on which the data will be sorted
        order (bool): Ascending or Descending order. Defaults to True.

    Returns:
        sorted_df (dataframe): sorted dataframe
    """
    sorted_df = df.sort_values(by=columns, ascending=order)
    sorted_df.reset_index(inplace= True, drop= True)
    return sorted_df

def select_data_year_wise(df, column, year):
    """This function is used to select data for a particular year

    Args:
        df (dataframe): Dataframe that has all the data
        column (str): Column with datetime value
        year (int): Year to be selected

    Returns:
        filtered_df (dataframe): Dataframe that has particular year data
    """
    filtered_df = df[df[column].dt.year == year]
    filtered_df.reset_index(inplace= True, drop= True)
    return filtered_df

def adjust_time_zone_offset(row, col):
    if row[col].date().year == 2020:
        if row[col].date() >= pd.Timestamp(2020, 3, 8) and row[col].date() <= pd.Timestamp(2020, 11, 1):
            return row[col] + timedelta(hours=1)
        else:
            return row[col]
    
    elif row[col].date().year == 2021:
        if row[col].date() >= pd.Timestamp(2021, 3, 14) and row[col].date() <= pd.Timestamp(2021, 11, 7):
            return row[col] + timedelta(hours=1)
        else:
            return row[col]
        
    elif row[col].date().year == 2022:
        if row[col].date() >= pd.Timestamp(2022, 3, 13) and row[col].date() <= pd.Timestamp(2022, 11, 6):
            return row[col] + timedelta(hours=1)
        else:
            return row[col]
        
    elif row[col].date().year == 2023:
        if row[col].date() >= pd.Timestamp(2022, 3, 12) and row[col].date() <= pd.Timestamp(2022, 11, 5):
            return row[col] + timedelta(hours=1)
        else:
            return row[col]

def adjust_time_zone(df):
    """This function is used ot adjust time based on time zone offset

    Args:
        df (datafrane): Dataframe without adjusted timeone

    Returns:
        df (datafrane): Dataframe with adjusted timeone
    """
    if df['time_zone_offset_y'].nunique() >= 2:
        if (abs(df['time_zone_offset_y'].value_counts().index[0]) == 1 + abs(df['time_zone_offset_y'].value_counts().index[1])) or \
            (abs(df['time_zone_offset_y'].value_counts().index[0]) + 1 == abs(df['time_zone_offset_y'].value_counts().index[1])):
            df['start_time'] = df.apply(lambda row: adjust_time_zone_offset(row, 'start_time'), axis=1)
            df['end_time'] = df.apply(lambda row: adjust_time_zone_offset(row, 'end_time'), axis=1)
            df['major_sleep_start'] = df.apply(lambda row: adjust_time_zone_offset(row, 'major_sleep_start'), axis=1)
            df['major_sleep_end'] = df.apply(lambda row: adjust_time_zone_offset(row, 'major_sleep_end'), axis=1)

    return df

def obtain_shift_instance(row):
    """This function is used to obtain shift instance based on shift start time

    Args:
        row (dataframe row): Dataframe row

    Returns:
        value (int): Shift instance indicator value
    """
    if row['off_day'] == 1:
        return 0
    else:
        if row['start_time'].hour + row['start_time'].minute / 60 >= 4 and row['start_time'].hour + row['start_time'].minute / 60 <= 17:
            return 1
        else:
            return -1
        
def obtain_last_day_of_day_shift_instance(df):
    """This function is used to obtain last day of day shift instance

    Args:
        df (dataframe): Dataframe before adding last day of day shift instance information

    Returns:
        df (dataframe): Dataframe after adding last day of day shift instance information
    """
    for row in range(df.shape[0] - 1):
        if df.loc[row, 'shift_instance'] == 0 or df.loc[row, 'shift_instance'] == -1:
            df.loc[row, 'last_day_of_day_shift'] = 0
        else:
            for next_row in range(row+1, df.shape[0] - 1):
                if df.loc[next_row, 'shift_instance'] == 1:
                    df.loc[row, 'last_day_of_day_shift'] = 0
                    break
                elif df.loc[next_row, 'shift_instance'] == -1:
                    df.loc[row, 'last_day_of_day_shift'] = 1
                    break
                next_row = next_row + 1
        row = row + 1

    df.loc[df.shape[0] - 1, 'last_day_of_day_shift'] = 0
    return df

def obtain_data_based_on_shift_instance_indicator(df, shift_instance):
    """Obtain data based on shift instance number

    Args:
        df (dataframe): Dataframe with all data
        shift_instance (int): shift instance indicator value

    Returns:
        df (dataframe): Filtered dataframe with all values for a particular shift instance
    """
    filtered_df = df[df['shift_instance'] == shift_instance]
    filtered_df.reset_index(inplace= True, drop= True)
    return filtered_df

def sin_conversion(row, column):
    """Convert value to sin based value

    Args:
        row (dataframe row): Row of dataframe
        column (dataframe column): Column of dataframe

    Returns:
        sin_value (float): Sin value
    """
    sin_value = np.sin(2 * np.pi * row[column] / 1440)
    return sin_value

def cos_conversion(row, column):
    """Convert value to cos based value

    Args:
        row (dataframe row): Row of dataframe
        column (dataframe column): Column of dataframe

    Returns:
        cos_value (float): Cos value
    """    
    cos_value = np.cos(2 * np.pi * row[column] / 1440)
    return cos_value

def drop_column(df, columns):
    """This function is used to drop the columns from dataframe

    Args:
        df (dataframe): Dataframe consisting column to be dropped
        columns (str or list): One or more names of columns which needs to be dropped

    Returns:
        filtered_df (dataframe): Dataframe after droppping the column(s)
    """
    filtered_df = df.drop(labels= columns, axis=1)
    return filtered_df

def min_max_scalar(df, useage, user_id, shift_instance, prev_delta_days, next_day_delta):
    """Scale data betweeen min and max values

    Args:
        df (dataframe): Unscaled dataframe
        useage (str): Model usage (either Sleep minutes or sleep start time)
        user_id (str): User id of the user's model
        shift_instance (str): shift instance of the model
        prev_delta_days (str): previous delta days used
        next_day_delta (int): next delta days used

    Returns:
        normalized_sleep_minutes_df (dataframe): Scaled dataframe
    """
    scalar = MinMaxScaler()
    scalar.fit(df)
    if not os.path.exists('./models/users/'+user_id+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/'):
        os.makedirs('./models/users/'+user_id+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/')
    joblib.dump(scalar, './models/users/'+user_id+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/scaler_model.joblib')
    scaled_data = scalar.transform(df)
    normalized_sleep_minutes_df = pd.DataFrame(scaled_data, columns=df.columns)
    return normalized_sleep_minutes_df

def load_scalar_model(useage, user_id, shift_instance, prev_delta_days, next_day_delta):
    """Load the scalar model

    Args:
        useage (str): Model usage (either Sleep minutes or sleep start time)
        user_id (str): User id of the user's model
        shift_instance (str): shift instance of the model
        prev_delta_days (str): previous delta days used
        next_day_delta (int): next delta days used
    """
    scalar = joblib.load('./models/users/'+user_id+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/scaler_model.joblib')
    return scalar

def train_test_split(df, label, test_size):
    """Split the data for train, test and validation

    Args:
        df (dataframe): dataframe that will be split between train, test and validation
        label (str): label to predict
        test_size (int): test size

    Returns:
        X_train (dataframe): Train attributes
        X_val (dataframe): Validation attributes
        X_test (dataframe): Test attributes
        y_train (series): Train label
        y_val (series): Validation label
        y_test (series): Test label
    """
    X = df.drop(label, axis=1)
    y = df[label]
    X_train, X_val, X_test = X.iloc[:test_size, :], X.iloc[test_size:, :], X.iloc[test_size:, :]
    y_train, y_val, y_test = y[:test_size], y[test_size:], y[test_size:]
    X_train.reset_index(inplace= True, drop= True)
    X_val.reset_index(inplace= True, drop= True)
    X_test.reset_index(inplace= True, drop= True)
    y_train.reset_index(inplace= True, drop= True)
    y_val.reset_index(inplace= True, drop= True)
    y_test.reset_index(inplace= True, drop= True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_dataset(df):
    """Convert dataframe or series to numpy array

    Args:
        df (dataframe or series): dataframe or series to be converted

    Returns:
        numpy_array (Numpy array): Converted to numpy array
    """
    numpy_array = df.to_numpy()
    return numpy_array

def save_model(useage, user_id, shift_instance, prev_delta_days, next_day_delta, monitor= 'val_loss', save_best_only=True, mode='min', verbose=1):
    """This function will save the model

    Args:
        useage (str): Model usage (either Sleep minutes or sleep start time)
        user_id (str): User id of the user's model
        shift_instance (str): shift instance of the model
        prev_delta_days (str): previous delta days used
        next_day_delta (int): next delta days used
        monitor (str): Parameter to choose to save the model. Defaults to 'val_loss'.
        save_best_only (bool): Check which model version to save. Defaults to True.
        mode (str): Comparision creteria. Defaults to 'min'.
        verbose (int): Training progress parameter. Defaults to 1.

    Returns:
        model checkpoint: model checkpoint
    """
    return tf.keras.callbacks.ModelCheckpoint(
    './models/users/'+user_id+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/model.keras',
    save_best_only=True,
    monitor=monitor,
    mode=mode,
    verbose=1
    )



def save_model_features(useage, user_id, shift_instance, prev_delta_days, next_day_delta, features):
    """This function will save the features used for modelling

    Args:
        useage (str): Model usage (either Sleep minutes or sleep start time)
        user_id (str): User id of the user's model
        shift_instance (str): shift instance of the model
        prev_delta_days (str): previous delta days used
        next_day_delta (int): next delta days used
        features (str): list of features 
    """
    with open('./models/users/'+user_id+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/features.txt', 'w') as f:
        f.write(features)

def load_model_features(useage, pattern, shift_instance, prev_delta_days, next_day_delta):
    with open('./models/patterns/'+pattern+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/features.txt', 'r') as f:
        features = f.readlines()
        pattern = re.compile(r'\w+')
        features = pattern.findall(features[0])
        return features

def load_model(useage, pattern, shift_instance, prev_delta_days, next_day_delta):
    """This function will load the model

    Args:
        useage (str): Model usage (either Sleep minutes or sleep start time)
        pattern (str): Pattern user's model
        shift_instance (str): shift instance of the model
        prev_delta_days (str): previous delta days used
        next_day_delta (int): next delta days used
    """
    model = tf.keras.models.load_model('./models/users/09bc946a99d02c3d63'+'/'+shift_instance+'/'+useage+'/'+str(prev_delta_days)+'_prev_days_'+str(next_day_delta)+'_next_days_model/model.keras')
    print(f"Model loaded from ./models/patterns/{pattern}/{shift_instance}/{useage}/{str(prev_delta_days)}_prev_days_{str(next_day_delta)}_next_days_model/model.keras")

    return model

def bound_predicted_sin_values(series):
    """This function is will predicted values of sleep start time, if the value is greater than 1 or less than -1, convert it to 1 ot -1 respectively

    Args:
        series (pandas series): Data with the predicted results

    Returns:
        series (pandas series): Data with the predicted results after conversion
    """
    series = np.where(series > 1, 1, series)
    series = np.where(series < -1 , -1, series)
    series = pd.Series(series)
    return series

def convert_minutes_from_sin_values(series):
    """Convert the sin based value to minutes

    Args:
        series (pandas dataframe): sin based value

    Returns:
        series (pandas dataframe): int value converted from sin based value
    """
    series = abs(round(pd.Series(np.arcsin(series) * 1440 / (2 * np.pi))))
    return series

def adjust_sleep_time_for_night_shift(series):
    """This function will adjust the shift time for night shift for the predicted values of sleep start time model

    Args:
        series (pandas series): data with predicted values

    Returns:
        series (pandas series): data with predicted values with adjustment
    """
    series = series + 720
    return series

def convert_to_hours(minutes):
    """Convert minutes to time value

    Args:
        minutes (int): time in minutes

    Returns:
        time_format (datetime): time value
    """
    if pd.notna(minutes) and isinstance(minutes, (int, float)):
        minutes = int(minutes)  # Convert to integer
        hours = minutes // 60
        minutes = minutes % 60
        time_str = f"{hours:02d}:{minutes:02d}"
        time_format = pd.to_datetime(time_str, format='%H:%M').time()
        return time_format