from common.config_constants import GENERATED_DATASETS_PATH, SLEEP_DAILY_DATASET_SUBPATH, SHIFT_INSTANCES_DATASET_SUBPATH, PREDICTION_START_DATE, \
                                    PREDICTION_TOTAL_DAYS_TO_PREDICT, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS, SHIFT_INSTANCE
from common.functions import merge_data, attribute_datatype_change_to_datetime, obtain_shift_instance, load_model, load_model_features, prepare_dataset, \
                             load_scalar_model, bound_predicted_sin_values, convert_minutes_from_sin_values, adjust_sleep_time_for_night_shift, \
                             save_csv_file, convert_to_hours
from main_extraction_pipeline import DataExtraction
from main_preprocessing_pipeline import DataPreprocessing
from main_training_pipeline import Training
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from common.functions import read_csv_file
from common.config_constants import USER_PROFILE_FEATURES
from pandas import merge
import tensorflow as tf


class Prediction:
    def __init__(self):
        self.data_preprocessing = DataPreprocessing()

    def prediction(self, user_id, user_shift_df, user_sleep_df):
        """Contains the prediction logic

        Args:
            user_id (str): user id of user who's prediction is to be made
            user_shift_df (dataframe): user's shift record
            user_sleep_df (dataframe): user's sleep record

        Returns:
            prediction_df (dataframe): predicted results
        """
        user_shift_df['start_time'] = attribute_datatype_change_to_datetime(user_shift_df, 'start_time')
        user_shift_df['shift_instance'] = user_shift_df.apply(lambda row: obtain_shift_instance(row), axis=1)

        prediction_df = pd.DataFrame(columns=['user_id', 'date', 'minutes_sleeping', 'sleep_start_time', 'shift_instance'])
        sleep_minutes_day_model = load_model('sleep_minutes', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_day_model = load_model('sleep_start_time', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_minutes_day_feature = load_model_features('sleep_minutes', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_day_feature = load_model_features('sleep_start_time', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_minutes_night_model = load_model('sleep_minutes', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_night_model = load_model('sleep_start_time', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_minutes_night_feature = load_model_features('sleep_minutes', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_night_feature = load_model_features('sleep_start_time', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)

        sleep_minutes_day_feature_with_label = sleep_minutes_day_feature.copy()
        sleep_minutes_day_feature_with_label.append('minutes_sleeping')
        sleep_start_time_day_feature_with_label = sleep_start_time_day_feature.copy()
        sleep_start_time_day_feature_with_label.append('sleep_start_time_minutes_sin')
        sleep_minutes_night_feature_with_label = sleep_minutes_night_feature.copy()
        sleep_minutes_night_feature_with_label.append('minutes_sleeping')
        sleep_start_time_night_feature_with_label = sleep_start_time_night_feature.copy()
        sleep_start_time_night_feature_with_label.append('sleep_start_time_minutes_sin')

        sleep_minutes_day_scalar_model = load_scalar_model('sleep_minutes', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_day_scalar_model = load_scalar_model('sleep_start_time', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_minutes_night_scalar_model = load_scalar_model('sleep_minutes', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_night_scalar_model = load_scalar_model('sleep_start_time', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        
        user_mapped_sleep_shift_df = self.data_preprocessing.map_sleep_shift_data(user_shift_df, user_sleep_df)
        user_mapped_sleep_shift_df['start_time'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'start_time')
        user_mapped_sleep_shift_df['end_time'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'end_time')

        user_mapped_sleep_shift_df['major_sleep_start'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'major_sleep_start')
        user_mapped_sleep_shift_df['major_sleep_end'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'major_sleep_end')
        user_mapped_sleep_shift_df['shift_date'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'shift_date')
        user_mapped_sleep_shift_df['shift_instance'] = user_mapped_sleep_shift_df.apply(lambda row: obtain_shift_instance(row), axis=1)

        user_mapped_sleep_shift_day_df = user_mapped_sleep_shift_df[user_mapped_sleep_shift_df['shift_instance'] == 1]
        user_mapped_sleep_shift_day_df.reset_index(inplace= True, drop= True)
        user_mapped_sleep_shift_night_df = user_mapped_sleep_shift_df[user_mapped_sleep_shift_df['shift_instance'] == -1]
        user_mapped_sleep_shift_night_df.reset_index(inplace= True, drop= True)
        
        user_mapped_sleep_shift_day_df['shift_date'] = user_mapped_sleep_shift_day_df['start_time'].dt.date
        user_mapped_sleep_shift_day_df['shift_date'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_day_df, 'shift_date')
        user_mapped_sleep_shift_night_df['shift_date'] = user_mapped_sleep_shift_night_df['start_time'].dt.date
        user_mapped_sleep_shift_night_df['shift_date'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_night_df, 'shift_date')
        
        user_mapped_sleep_shift_day_df = self.data_preprocessing.convert_time_to_sin_cos_values(user_mapped_sleep_shift_day_df)
        user_mapped_sleep_shift_night_df = self.data_preprocessing.convert_time_to_sin_cos_values(user_mapped_sleep_shift_night_df)
        user_mapped_sleep_shift_sleep_minutes_day_df, user_mapped_sleep_shift_sleep_start_time_day_df = self.data_preprocessing.transform_data_for_delta_days(user_mapped_sleep_shift_day_df)
        user_mapped_sleep_shift_sleep_minutes_night_df, user_mapped_sleep_shift_sleep_start_time_night_df = self.data_preprocessing.transform_data_for_delta_days(user_mapped_sleep_shift_night_df)
        
        user_mapped_sleep_shift_sleep_minutes_day_df['minutes_sleeping'] = 0
        user_mapped_sleep_shift_sleep_minutes_night_df['minutes_sleeping'] = 0
        user_mapped_sleep_shift_sleep_start_time_day_df['sleep_start_time_minutes_sin'] = 0
        user_mapped_sleep_shift_sleep_start_time_night_df['sleep_start_time_minutes_sin'] = 0

        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = sleep_minutes_day_scalar_model.transform(user_mapped_sleep_shift_sleep_minutes_day_df[sleep_minutes_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_day_df, columns= sleep_minutes_day_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = sleep_start_time_day_scalar_model.transform(user_mapped_sleep_shift_sleep_start_time_day_df[sleep_start_time_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_day_df, columns= sleep_start_time_day_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = sleep_minutes_night_scalar_model.transform(user_mapped_sleep_shift_sleep_minutes_night_df[sleep_minutes_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_night_df, columns= sleep_minutes_night_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = sleep_start_time_night_scalar_model.transform(user_mapped_sleep_shift_sleep_start_time_night_df[sleep_start_time_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_night_df, columns= sleep_start_time_night_feature_with_label)

        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = sleep_minutes_day_scalar_model.transform(user_mapped_sleep_shift_sleep_minutes_day_df[sleep_minutes_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_day_df, columns= sleep_minutes_day_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_minutes_day_df['shift_date'] = user_mapped_sleep_shift_sleep_minutes_day_df['shift_date']
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = sleep_minutes_night_scalar_model.transform(user_mapped_sleep_shift_sleep_minutes_night_df[sleep_minutes_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_night_df, columns= sleep_minutes_night_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df['shift_date'] = user_mapped_sleep_shift_sleep_minutes_night_df['shift_date']

        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = sleep_start_time_day_scalar_model.transform(user_mapped_sleep_shift_sleep_start_time_day_df[sleep_start_time_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_day_df, columns= sleep_start_time_day_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df['shift_date'] = user_mapped_sleep_shift_sleep_start_time_day_df['shift_date']
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = sleep_start_time_night_scalar_model.transform(user_mapped_sleep_shift_sleep_start_time_night_df[sleep_start_time_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_night_df, columns= sleep_start_time_night_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df['shift_date'] = user_mapped_sleep_shift_sleep_start_time_night_df['shift_date']

        prediction_start_date = datetime.strptime(PREDICTION_START_DATE, '%Y-%m-%d').date()
        user_mapped_sleep_shift_offday_df = user_mapped_sleep_shift_df[user_mapped_sleep_shift_df['off_day'] == 1]
        user_mapped_sleep_shift_offday_df = user_mapped_sleep_shift_offday_df[user_mapped_sleep_shift_offday_df['start_time'].dt.date < prediction_start_date]
        place_holder_minutes_sleeping = float(user_mapped_sleep_shift_offday_df.loc[user_mapped_sleep_shift_offday_df.index[-1], 'minutes_sleeping'])
        place_holder_sleep_start_time = user_mapped_sleep_shift_offday_df.loc[user_mapped_sleep_shift_offday_df.index[-1], 'major_sleep_start']
        place_holder_sleep_start_time = float(place_holder_sleep_start_time.time().minute + place_holder_sleep_start_time.time().hour * 60)

        for day in range(PREDICTION_TOTAL_DAYS_TO_PREDICT):
            date = prediction_start_date + timedelta(days=day)
            shift_instance = (user_mapped_sleep_shift_df[user_mapped_sleep_shift_df['start_time'].dt.date == date]['shift_instance']).tolist()
            if shift_instance:
                if shift_instance[0] == 1:
                    sleep_minutes_df = scaled_user_mapped_sleep_shift_sleep_minutes_day_df[scaled_user_mapped_sleep_shift_sleep_minutes_day_df['shift_date'].dt.date == date][sleep_minutes_day_feature]
                    sleep_minutes_df.reset_index(inplace= True, drop= True)
                    sleep_minutes_arr = prepare_dataset(sleep_minutes_df.loc[0, :])
                    sleep_minutes_arr = sleep_minutes_arr.reshape((1, len(sleep_minutes_arr)))
                    sleep_minutes_df['predicted_sleep_minutes'] = sleep_minutes_day_model.predict(sleep_minutes_arr)
                    sleep_minutes_columns = sleep_minutes_df.columns
                    sleep_minutes_df = sleep_minutes_day_scalar_model.inverse_transform(sleep_minutes_df)
                    sleep_minutes_df = pd.DataFrame(sleep_minutes_df, columns=sleep_minutes_columns)
                    minutes_sleeping = sleep_minutes_df['predicted_sleep_minutes'].values[0]
                    sleep_start_time_df = scaled_user_mapped_sleep_shift_sleep_start_time_day_df[scaled_user_mapped_sleep_shift_sleep_start_time_day_df['shift_date'].dt.date == date][sleep_start_time_day_feature]
                    sleep_start_time_df.reset_index(inplace= True, drop= True)
                    sleep_start_time_arr = prepare_dataset(sleep_start_time_df.loc[0, :])
                    sleep_start_time_arr = sleep_start_time_arr.reshape((1, len(sleep_start_time_arr)))
                    sleep_start_time_df['predicted_sleep_start_time'] = sleep_start_time_day_model.predict(sleep_start_time_arr)
                    sleep_start_time_columns = sleep_start_time_df.columns
                    sleep_start_time_df = sleep_start_time_day_scalar_model.inverse_transform(sleep_start_time_df)
                    sleep_start_time_df = pd.DataFrame(sleep_start_time_df, columns=sleep_start_time_columns)
                    sleep_start_time_df['predicted_sleep_start_time'] = bound_predicted_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
                    sleep_start_time_df['predicted_sleep_start_time'] = convert_minutes_from_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
                    sleep_start_time = sleep_start_time_df['predicted_sleep_start_time'].values[0]

                elif shift_instance[0] == -1:
                    sleep_minutes_df = scaled_user_mapped_sleep_shift_sleep_minutes_night_df[scaled_user_mapped_sleep_shift_sleep_minutes_night_df['shift_date'].dt.date == date][sleep_minutes_night_feature]
                    sleep_minutes_df.reset_index(inplace= True, drop= True)
                    sleep_minutes_arr = prepare_dataset(sleep_minutes_df.loc[0, :])
                    sleep_minutes_arr = sleep_minutes_arr.reshape((1, len(sleep_minutes_arr)))
                    sleep_minutes_df['predicted_sleep_minutes'] = sleep_minutes_night_model.predict(sleep_minutes_arr)
                    sleep_minutes_columns = sleep_minutes_df.columns
                    sleep_minutes_df = sleep_minutes_night_scalar_model.inverse_transform(sleep_minutes_df)
                    sleep_minutes_df = pd.DataFrame(sleep_minutes_df, columns=sleep_minutes_columns)
                    minutes_sleeping= sleep_minutes_df['predicted_sleep_minutes'].values[0]
                    sleep_start_time_df = scaled_user_mapped_sleep_shift_sleep_start_time_night_df[scaled_user_mapped_sleep_shift_sleep_start_time_night_df['shift_date'].dt.date == date][sleep_start_time_night_feature]
                    sleep_start_time_df.reset_index(inplace= True, drop= True)
                    sleep_start_time_arr = prepare_dataset(sleep_start_time_df.loc[0, :])
                    sleep_start_time_arr = sleep_start_time_arr.reshape((1, len(sleep_start_time_arr)))
                    sleep_start_time_df['predicted_sleep_start_time'] = sleep_start_time_night_model.predict(sleep_start_time_arr)
                    sleep_start_time_columns = sleep_start_time_df.columns
                    sleep_start_time_df = sleep_start_time_night_scalar_model.inverse_transform(sleep_start_time_df)
                    sleep_start_time_df = pd.DataFrame(sleep_start_time_df, columns=sleep_start_time_columns)
                    sleep_start_time_df['predicted_sleep_start_time'] = bound_predicted_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
                    sleep_start_time_df['predicted_sleep_start_time'] = convert_minutes_from_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
                    sleep_start_time_df['predicted_sleep_start_time'] = adjust_sleep_time_for_night_shift(sleep_start_time_df['predicted_sleep_start_time'])
                    sleep_start_time = sleep_start_time_df['predicted_sleep_start_time'].values[0]
                    
                else:
                    minutes_sleeping = place_holder_minutes_sleeping
                    sleep_start_time = place_holder_sleep_start_time
            else:
                minutes_sleeping = np.NaN
                sleep_start_time = np.NaN
                shift_instance = [np.NaN]

            prediction_df.loc[day, 'user_id'] = user_id
            prediction_df.loc[day, 'date'] = date
            prediction_df.loc[day, 'minutes_sleeping'] = minutes_sleeping
            prediction_df.loc[day, 'sleep_start_time'] = sleep_start_time
            prediction_df.loc[day, 'shift_instance'] = shift_instance[0]

        return prediction_df
    
    def prediction_day(self, user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df, user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df, pattern, user_id):

        minutes_sleeping_predictions = []
        sleep_start_time_predictions = []

        sleep_minutes_day_model = load_model('sleep_minutes', pattern, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_day_model = load_model('sleep_start_time', pattern, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_minutes_day_feature = load_model_features('sleep_minutes', pattern, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_day_feature = load_model_features('sleep_start_time', pattern, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)

        sleep_minutes_day_feature_with_label = sleep_minutes_day_feature.copy()
        sleep_minutes_day_feature_with_label.append('minutes_sleeping')
        sleep_start_time_day_feature_with_label = sleep_start_time_day_feature.copy()
        sleep_start_time_day_feature_with_label.append('sleep_start_time_minutes_sin')

        sleep_minutes_day_scalar_model = load_scalar_model('sleep_minutes', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_day_scalar_model = load_scalar_model('sleep_start_time', user_id, SHIFT_INSTANCE[0], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)

        user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df['minutes_sleeping'] = 0
        user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df['sleep_start_time_minutes_sin'] = 0

        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = sleep_minutes_day_scalar_model.transform(user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df[sleep_minutes_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_day_df, columns= sleep_minutes_day_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = sleep_start_time_day_scalar_model.transform(user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df[sleep_start_time_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_day_df, columns= sleep_start_time_day_feature_with_label)

        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = sleep_minutes_day_scalar_model.transform(user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df[sleep_minutes_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_day_df, columns= sleep_minutes_day_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_minutes_day_df['shift_date'] = user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df['shift_date']

        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = sleep_start_time_day_scalar_model.transform(user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df[sleep_start_time_day_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_day_df, columns= sleep_start_time_day_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_day_df['shift_date'] = user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df['shift_date']

        for day in range(scaled_user_mapped_sleep_shift_sleep_minutes_day_df.shape[0]):
            date = scaled_user_mapped_sleep_shift_sleep_minutes_day_df.loc[day, 'shift_date']
            sleep_minutes_df = scaled_user_mapped_sleep_shift_sleep_minutes_day_df[scaled_user_mapped_sleep_shift_sleep_minutes_day_df['shift_date'] == date][sleep_minutes_day_feature]
            sleep_minutes_df.reset_index(inplace= True, drop= True)
            sleep_minutes_arr = prepare_dataset(sleep_minutes_df.loc[0, :])
            sleep_minutes_arr = sleep_minutes_arr.reshape((1, len(sleep_minutes_arr)))
            sleep_minutes_df['predicted_sleep_minutes'] = sleep_minutes_day_model.predict(sleep_minutes_arr)
            sleep_minutes_columns = sleep_minutes_df.columns
            sleep_minutes_df = sleep_minutes_day_scalar_model.inverse_transform(sleep_minutes_df)
            sleep_minutes_df = pd.DataFrame(sleep_minutes_df, columns=sleep_minutes_columns)
            minutes_sleeping = sleep_minutes_df['predicted_sleep_minutes'].values[0]
            sleep_start_time_df = scaled_user_mapped_sleep_shift_sleep_start_time_day_df[scaled_user_mapped_sleep_shift_sleep_start_time_day_df['shift_date'] == date][sleep_start_time_day_feature]
            sleep_start_time_df.reset_index(inplace= True, drop= True)
            sleep_start_time_arr = prepare_dataset(sleep_start_time_df.loc[0, :])
            sleep_start_time_arr = sleep_start_time_arr.reshape((1, len(sleep_start_time_arr)))
            sleep_start_time_df['predicted_sleep_start_time'] = sleep_start_time_day_model.predict(sleep_start_time_arr)
            sleep_start_time_columns = sleep_start_time_df.columns
            sleep_start_time_df = sleep_start_time_day_scalar_model.inverse_transform(sleep_start_time_df)
            sleep_start_time_df = pd.DataFrame(sleep_start_time_df, columns=sleep_start_time_columns)
            sleep_start_time_df['predicted_sleep_start_time'] = bound_predicted_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
            sleep_start_time_df['predicted_sleep_start_time'] = convert_minutes_from_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
            sleep_start_time_df['predicted_sleep_start_time'] = adjust_sleep_time_for_night_shift(sleep_start_time_df['predicted_sleep_start_time'])
            sleep_start_time = sleep_start_time_df['predicted_sleep_start_time'].values[0]
            minutes_sleeping_predictions.append(minutes_sleeping)
            sleep_start_time_predictions.append(sleep_start_time)

        return minutes_sleeping_predictions, sleep_start_time_predictions
    

    def prediction_night(self, user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df, user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df, pattern, user_id):
        minutes_sleeping_predictions = []
        sleep_start_time_predictions = []

        sleep_minutes_night_model = load_model('sleep_minutes', pattern, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_night_model = load_model('sleep_start_time', pattern, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_minutes_night_feature = load_model_features('sleep_minutes', pattern, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_night_feature = load_model_features('sleep_start_time', pattern, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)

        sleep_minutes_night_feature_with_label = sleep_minutes_night_feature.copy()
        sleep_minutes_night_feature_with_label.append('minutes_sleeping')
        sleep_start_time_night_feature_with_label = sleep_start_time_night_feature.copy()
        sleep_start_time_night_feature_with_label.append('sleep_start_time_minutes_sin')

        sleep_minutes_night_scalar_model = load_scalar_model('sleep_minutes', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        sleep_start_time_night_scalar_model = load_scalar_model('sleep_start_time', user_id, SHIFT_INSTANCE[1], PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        
        user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df['minutes_sleeping'] = 0
        user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df['sleep_start_time_minutes_sin'] = 0

        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = sleep_minutes_night_scalar_model.transform(user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df[sleep_minutes_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_night_df, columns= sleep_minutes_night_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = sleep_start_time_night_scalar_model.transform(user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df[sleep_start_time_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_night_df, columns= sleep_start_time_night_feature_with_label)

        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = sleep_minutes_night_scalar_model.transform(user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df[sleep_minutes_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_minutes_night_df, columns= sleep_minutes_night_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_minutes_night_df['shift_date'] = user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df['shift_date']

        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = sleep_start_time_night_scalar_model.transform(user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df[sleep_start_time_night_feature_with_label])
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df = pd.DataFrame(scaled_user_mapped_sleep_shift_sleep_start_time_night_df, columns= sleep_start_time_night_feature_with_label)
        scaled_user_mapped_sleep_shift_sleep_start_time_night_df['shift_date'] = user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df['shift_date']

        for night in range(scaled_user_mapped_sleep_shift_sleep_minutes_night_df.shape[0]):
            date = scaled_user_mapped_sleep_shift_sleep_minutes_night_df.loc[night, 'shift_date']
            sleep_minutes_df = scaled_user_mapped_sleep_shift_sleep_minutes_night_df[scaled_user_mapped_sleep_shift_sleep_minutes_night_df['shift_date'] == date][sleep_minutes_night_feature]
            sleep_minutes_df.reset_index(inplace= True, drop= True)
            sleep_minutes_arr = prepare_dataset(sleep_minutes_df.loc[0, :])
            sleep_minutes_arr = sleep_minutes_arr.reshape((1, len(sleep_minutes_arr)))
            sleep_minutes_df['predicted_sleep_minutes'] = sleep_minutes_night_model.predict(sleep_minutes_arr)
            sleep_minutes_columns = sleep_minutes_df.columns
            sleep_minutes_df = sleep_minutes_night_scalar_model.inverse_transform(sleep_minutes_df)
            sleep_minutes_df = pd.DataFrame(sleep_minutes_df, columns=sleep_minutes_columns)
            minutes_sleeping = sleep_minutes_df['predicted_sleep_minutes'].values[0]
            sleep_start_time_df = scaled_user_mapped_sleep_shift_sleep_start_time_night_df[scaled_user_mapped_sleep_shift_sleep_start_time_night_df['shift_date'] == date][sleep_start_time_night_feature]
            sleep_start_time_df.reset_index(inplace= True, drop= True)
            sleep_start_time_arr = prepare_dataset(sleep_start_time_df.loc[0, :])
            sleep_start_time_arr = sleep_start_time_arr.reshape((1, len(sleep_start_time_arr)))
            sleep_start_time_df['predicted_sleep_start_time'] = sleep_start_time_night_model.predict(sleep_start_time_arr)
            sleep_start_time_columns = sleep_start_time_df.columns
            sleep_start_time_df = sleep_start_time_night_scalar_model.inverse_transform(sleep_start_time_df)
            sleep_start_time_df = pd.DataFrame(sleep_start_time_df, columns=sleep_start_time_columns)
            sleep_start_time_df['predicted_sleep_start_time'] = bound_predicted_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
            sleep_start_time_df['predicted_sleep_start_time'] = convert_minutes_from_sin_values(sleep_start_time_df['predicted_sleep_start_time'])
            sleep_start_time = sleep_start_time_df['predicted_sleep_start_time'].values[0]
            minutes_sleeping_predictions.append(minutes_sleeping)
            sleep_start_time_predictions.append(sleep_start_time)

        return minutes_sleeping_predictions, sleep_start_time_predictions

def main():
    data_extraction = DataExtraction()
    data_preprocessing = DataPreprocessing()
    input = data_extraction.read_input()
    user_ids = data_extraction.extract_user_id_from_input(input)
    prediction = Prediction()

    for user_id in user_ids:
        try:
            user_mapped_sleep_shift_data_all_year_df = read_csv_file(f'./data/generated/users/{user_id}/sleep-shift-mapped-data/all-years/{user_id}_sleep_shift_mapped_data_all_year_data.csv', 0)
            user_profile = read_csv_file(f'./data/generated/users/{user_id}/user_profile.csv', 0)
            user_profile_df = user_profile[USER_PROFILE_FEATURES]
            user_mapped_sleep_shift_df = merge(user_mapped_sleep_shift_data_all_year_df, user_profile_df, on='shift_date', how='left')
            user_mapped_sleep_shift_df.drop(columns=['user_id_y'], inplace=True)
            user_mapped_sleep_shift_df.rename(columns={'user_id_x': 'user_id'}, inplace=True)

            user_mapped_sleep_shift_all_year_day_df, user_mapped_sleep_shift_all_year_night_df, user_mapped_sleep_shift_all_year_off_day_df = data_preprocessing.divide_data_based_on_shift_instance(user_mapped_sleep_shift_df)
            user_mapped_sleep_shift_all_year_day_df = data_preprocessing.convert_time_to_sin_cos_values(user_mapped_sleep_shift_all_year_day_df)
            user_mapped_sleep_shift_all_year_night_df = data_preprocessing.convert_time_to_sin_cos_values(user_mapped_sleep_shift_all_year_night_df)
            user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df, user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df = data_preprocessing.transform_data_for_delta_days(user_mapped_sleep_shift_all_year_day_df)
            user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df, user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df = data_preprocessing.transform_data_for_delta_days(user_mapped_sleep_shift_all_year_night_df)

            pattern = None 
            pattern_df = read_csv_file('./data/raw/user-pattern/user_pattern_big5.csv', 0)
            user_pattern_df = pattern_df[pattern_df['hashed_user_id'] == user_id]

            if user_pattern_df.shape[0] > 0:
                user_pattern_df.reset_index(inplace= True, drop= True)
                pattern = user_pattern_df.loc[0, 'pattern']
                if pattern:
                    if user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df.shape[0] > 0 and user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df.shape[0] > 0:
                        minutes_sleeping_predictions, sleep_start_time_predictions = prediction.prediction_day(user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df, user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df, pattern, user_id)
                        user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df['minutes_sleeping_predictions'] = minutes_sleeping_predictions
                        user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df['sleep_start_time_predictions'] = sleep_start_time_predictions
                        save_csv_file(user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df, f'./data/generated/users/{user_id}/{SHIFT_INSTANCE[0]}_{PREVIOUS_DELTA_DAYS}_PDD_{NEXT_DELTA_DAYS}_NDD_result.csv')        
                        print(f'Result file stored at location: ./data/generated/users/{user_id}/{SHIFT_INSTANCE[0]}_{PREVIOUS_DELTA_DAYS}_PDD_{NEXT_DELTA_DAYS}_NDD_result.csv')
                    else:
                        print(f'Data of Day Shift modelling not available for {user_id} user')
                        
                    if user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df.shape[0] > 0 and user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df.shape[0] > 0:
                        minutes_sleeping_predictions, sleep_start_time_predictions = prediction.prediction_night(user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df, user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df, pattern, user_id)
                        user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df['minutes_sleeping_predictions'] = minutes_sleeping_predictions
                        user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df['sleep_start_time_predictions'] = sleep_start_time_predictions
                        save_csv_file(user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df, f'./data/generated/users/{user_id}/{SHIFT_INSTANCE[1]}_{PREVIOUS_DELTA_DAYS}_PDD_{NEXT_DELTA_DAYS}_NDD_result.csv')
                        print(f'Result file stored at location: ./data/generated/users/{user_id}/{SHIFT_INSTANCE[1]}_{PREVIOUS_DELTA_DAYS}_PDD_{NEXT_DELTA_DAYS}_NDD_result.csv')
                    else:
                        print(f'Data of Night Shift modelling not available for {user_id} user')
                else:
                    print(f'Work shift pattern not available for {user_id} user')    
            else:
                print(f'Work shift pattern not available for {user_id} user')
            
        except:
            print(f'Could not produce result for {user_id}. Please remove the user from input_user_list.txt file.')

if __name__ == "__main__":
    main()