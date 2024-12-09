from pandas import merge
from datetime import datetime, timedelta
from common.config_constants import INPUT_USER_FILE, GENERATED_DATASETS_PATH, SLEEP_DAILY_DATASET_SUBPATH, SHIFT_INSTANCES_DATASET_SUBPATH, USER_SHIFT_START_END_FEATURES, USER_SLEEP_START_END_FEATURES, PREVIOUS_DELTA_DAYS, \
                                    NEXT_DELTA_DAYS, USER_SLEEP_SHIFT_TRANSFORMATION_FEATURES, MODEL_FIXED_FEATURES, USER_PROFILE_FEATURES
from common.functions import read_text_file, extract_user_id, read_csv_file, attribute_datatype_change_to_datetime, adjust_shift_start_time_for_off_days, drop_null_values, sort_values, remove_duplicate_values, \
                            select_data_year_wise, adjust_time_zone, merge_data, obtain_shift_instance, obtain_last_day_of_day_shift_instance, obtain_data_based_on_shift_instance_indicator, sin_conversion, cos_conversion
from main_extraction_pipeline import DataExtraction

class DataPreprocessing:

    def divide_data_based_on_shift_instance(self, user_mapped_sleep_shift_df):
        """This function will divide the data based on the shift timings into day shift, night shift and off days.

        Args:
            user_mapped_sleep_shift_df (dataframe): Sleep shift mapped data for the user 

        Returns:
            user_mapped_sleep_shift_all_year_day_df (dataframe): Sleep shift mapped data for the user (during day shift)
            user_mapped_sleep_shift_all_year_night_df (dataframe): Sleep shift mapped data for the user (during night shift)
            user_mapped_sleep_shift_all_year_off_day_df (dataframe): Sleep shift mapped data for the user (during offday shift)
        """
        user_mapped_sleep_shift_df['shift_date'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'shift_date', errors='coerce')
        user_mapped_sleep_shift_df['start_time'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'start_time', errors='coerce')
        user_mapped_sleep_shift_df['end_time'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'end_time', errors='coerce')
        user_mapped_sleep_shift_df['major_sleep_start'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'major_sleep_start', errors='coerce')
        user_mapped_sleep_shift_df['major_sleep_end'] = attribute_datatype_change_to_datetime(user_mapped_sleep_shift_df, 'major_sleep_end', errors='coerce')

        user_mapped_sleep_shift_2020_df = select_data_year_wise(user_mapped_sleep_shift_df, 'start_time', 2020)
        user_mapped_sleep_shift_2021_df = select_data_year_wise(user_mapped_sleep_shift_df, 'start_time', 2021)
        user_mapped_sleep_shift_2022_df = select_data_year_wise(user_mapped_sleep_shift_df, 'start_time', 2022)
        user_mapped_sleep_shift_2023_df = select_data_year_wise(user_mapped_sleep_shift_df, 'start_time', 2023)

        adjusted_timezone_user_mapped_sleep_shift_2020_df = adjust_time_zone(user_mapped_sleep_shift_2020_df)
        adjusted_timezone_user_mapped_sleep_shift_2021_df = adjust_time_zone(user_mapped_sleep_shift_2021_df)
        adjusted_timezone_user_mapped_sleep_shift_2022_df = adjust_time_zone(user_mapped_sleep_shift_2022_df)
        adjusted_timezone_user_mapped_sleep_shift_2023_df = adjust_time_zone(user_mapped_sleep_shift_2023_df)

        adjusted_timezone_user_mapped_sleep_shift_df = merge_data(adjusted_timezone_user_mapped_sleep_shift_2020_df, adjusted_timezone_user_mapped_sleep_shift_2021_df, \
                                                                adjusted_timezone_user_mapped_sleep_shift_2022_df, adjusted_timezone_user_mapped_sleep_shift_2023_df)
        
        user_mapped_sleep_shift_all_year_df = adjusted_timezone_user_mapped_sleep_shift_df.copy()
        user_mapped_sleep_shift_all_year_df['shift_instance'] = user_mapped_sleep_shift_all_year_df.apply(lambda row: obtain_shift_instance(row), axis=1)
        user_mapped_sleep_shift_all_year_df = obtain_last_day_of_day_shift_instance(user_mapped_sleep_shift_all_year_df)

        user_mapped_sleep_shift_all_year_day_df = obtain_data_based_on_shift_instance_indicator(user_mapped_sleep_shift_all_year_df, 1)
        user_mapped_sleep_shift_all_year_night_df = obtain_data_based_on_shift_instance_indicator(user_mapped_sleep_shift_all_year_df, -1)
        user_mapped_sleep_shift_all_year_off_day_df = obtain_data_based_on_shift_instance_indicator(user_mapped_sleep_shift_all_year_df, 0)

        return user_mapped_sleep_shift_all_year_day_df, user_mapped_sleep_shift_all_year_night_df, user_mapped_sleep_shift_all_year_off_day_df

    def convert_time_to_sin_cos_values(self, user_mapped_sleep_shift_all_year_shift_instance_df):
        """Convert the time to sin cos based values

        Args:
            user_mapped_sleep_shift_all_year_shift_instance_df (dataframe): dataframe with time data

        Returns:
            user_mapped_sleep_shift_all_year_shift_instance_df (dataframe): dataframe with time data converted to sin cos based data
        """
        user_mapped_sleep_shift_all_year_shift_instance_df['shift_start_time_minutes'] = user_mapped_sleep_shift_all_year_shift_instance_df['start_time'].dt.hour * 60 + user_mapped_sleep_shift_all_year_shift_instance_df['start_time'].dt.minute
        user_mapped_sleep_shift_all_year_shift_instance_df['shift_end_time_minutes'] = user_mapped_sleep_shift_all_year_shift_instance_df['end_time'].dt.hour * 60 + user_mapped_sleep_shift_all_year_shift_instance_df['end_time'].dt.minute
        user_mapped_sleep_shift_all_year_shift_instance_df['sleep_start_time_minutes'] = user_mapped_sleep_shift_all_year_shift_instance_df['major_sleep_start'].dt.hour * 60 + user_mapped_sleep_shift_all_year_shift_instance_df['major_sleep_start'].dt.minute
        user_mapped_sleep_shift_all_year_shift_instance_df['sleep_end_time_minutes'] = user_mapped_sleep_shift_all_year_shift_instance_df['major_sleep_end'].dt.hour * 60 + user_mapped_sleep_shift_all_year_shift_instance_df['major_sleep_end'].dt.minute

        user_mapped_sleep_shift_all_year_shift_instance_df['shift_start_time_minutes_sin'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: sin_conversion(row, 'shift_start_time_minutes'), axis=1)
        user_mapped_sleep_shift_all_year_shift_instance_df['shift_start_time_minutes_cos'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: cos_conversion(row, 'shift_start_time_minutes'), axis=1)
        user_mapped_sleep_shift_all_year_shift_instance_df['shift_end_time_minutes_sin'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: sin_conversion(row, 'shift_end_time_minutes'), axis=1)
        user_mapped_sleep_shift_all_year_shift_instance_df['shift_end_time_minutes_cos'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: cos_conversion(row, 'shift_end_time_minutes'), axis=1)
        user_mapped_sleep_shift_all_year_shift_instance_df['sleep_start_time_minutes_sin'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: sin_conversion(row, 'sleep_start_time_minutes'), axis=1)
        user_mapped_sleep_shift_all_year_shift_instance_df['sleep_start_time_minutes_cos'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: cos_conversion(row, 'sleep_start_time_minutes'), axis=1)
        user_mapped_sleep_shift_all_year_shift_instance_df['sleep_end_time_minutes_sin'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: sin_conversion(row, 'sleep_end_time_minutes'), axis=1)
        user_mapped_sleep_shift_all_year_shift_instance_df['sleep_end_time_minutes_cos'] = user_mapped_sleep_shift_all_year_shift_instance_df.apply(lambda row: cos_conversion(row, 'sleep_end_time_minutes'), axis=1)
        
        return user_mapped_sleep_shift_all_year_shift_instance_df

    def transform_data_for_delta_days(self, user_mapped_sleep_shift_all_year_df):
        """Create features based on the past days of sleep and shift data and future days of shift data

        Args:
            user_mapped_sleep_shift_all_year_df (dataframe): data from which features will be extracted

        Returns:
            user_mapped_sleep_shift_all_year_df[list(minutes_sleeping_features)]: data for sleep minutes modelling
            user_mapped_sleep_shift_all_year_df[list(sleep_start_time_features)]: data for sleep start time modelling
        """
        
        minutes_sleeping_features = set()
        sleep_start_time_features = set()
        # user_mapped_sleep_shift_all_year_df = user_mapped_sleep_shift_all_year_df[USER_SLEEP_SHIFT_TRANSFORMATION_FEATURES]
    
        if PREVIOUS_DELTA_DAYS > 0:
            for idx in range(PREVIOUS_DELTA_DAYS, user_mapped_sleep_shift_all_year_df.shape[0]):
                for day in range(1, PREVIOUS_DELTA_DAYS+1):
                    if user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_date'].date() - timedelta(days=day)  == user_mapped_sleep_shift_all_year_df.loc[idx-day, 'shift_date'].date():
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'off_day_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'off_day']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'minutes_of_duty_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'minutes_of_duty']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'streak_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'streak']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'minutes_resting_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'minutes_resting']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'minutes_sleeping_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'minutes_sleeping']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'minutes_sleep_latency_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'minutes_sleep_latency']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'wake_episodes_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'wake_episodes']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'mean_wake_episodes_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'mean_wake_episodes']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_start_time_minutes_sin_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'shift_start_time_minutes_sin']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_start_time_minutes_cos_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'shift_start_time_minutes_cos']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_end_time_minutes_sin_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'shift_end_time_minutes_sin']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_end_time_minutes_cos_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'shift_end_time_minutes_cos']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'sleep_start_minutes_sin_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'sleep_start_time_minutes_sin']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'sleep_start_minutes_cos_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'sleep_start_time_minutes_cos']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'sleep_end_minutes_sin_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'sleep_end_time_minutes_sin']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'sleep_end_minutes_cos_prev_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx-day, 'sleep_end_time_minutes_cos']

                        minutes_sleeping_features.add('off_day_prev_'+str(day))
                        minutes_sleeping_features.add('minutes_of_duty_prev_'+str(day))
                        minutes_sleeping_features.add('minutes_sleeping_prev_'+str(day))
                        minutes_sleeping_features.add('shift_start_time_minutes_sin_prev_'+str(day))
                        minutes_sleeping_features.add('shift_start_time_minutes_cos_prev_'+str(day))
                        minutes_sleeping_features.add('shift_end_time_minutes_sin_prev_'+str(day))
                        minutes_sleeping_features.add('shift_end_time_minutes_cos_prev_'+str(day))
                        minutes_sleeping_features.add('sleep_start_minutes_sin_prev_'+str(day))
                        minutes_sleeping_features.add('sleep_start_minutes_cos_prev_'+str(day))
                        minutes_sleeping_features.add('sleep_end_minutes_sin_prev_'+str(day))
                        minutes_sleeping_features.add('sleep_end_minutes_cos_prev_'+str(day))

                        sleep_start_time_features.add('off_day_prev_'+str(day))
                        sleep_start_time_features.add('minutes_of_duty_prev_'+str(day))
                        sleep_start_time_features.add('minutes_sleeping_prev_'+str(day))
                        sleep_start_time_features.add('shift_start_time_minutes_sin_prev_'+str(day))
                        sleep_start_time_features.add('shift_start_time_minutes_cos_prev_'+str(day))
                        sleep_start_time_features.add('shift_end_time_minutes_sin_prev_'+str(day))
                        sleep_start_time_features.add('shift_end_time_minutes_cos_prev_'+str(day))
                        sleep_start_time_features.add('sleep_start_minutes_sin_prev_'+str(day))
                        sleep_start_time_features.add('sleep_start_minutes_cos_prev_'+str(day))
                        sleep_start_time_features.add('sleep_end_minutes_sin_prev_'+str(day))
                        sleep_start_time_features.add('sleep_end_minutes_cos_prev_'+str(day))

        # user_mapped_sleep_shift_all_year_df = drop_null_values(user_mapped_sleep_shift_all_year_df).copy()

        if NEXT_DELTA_DAYS > 0:
            for idx in range(user_mapped_sleep_shift_all_year_df.shape[0] - NEXT_DELTA_DAYS):
                for day in range(1, NEXT_DELTA_DAYS+1):
                    if user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_date'].date() + datetime.timedelta(days=day)  == user_mapped_sleep_shift_all_year_df.loc[idx+1, 'shift_date'].date():
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'off_day_next_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx+1, 'off_day']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'minutes_of_duty_next_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx+1, 'minutes_of_duty']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'streak_next_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx+1, 'streak']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_start_time_minutes_sin_next_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx+1, 'shift_start_time_minutes_cos']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_start_time_minutes_cos_next_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx+1, 'shift_start_time_minutes_cos']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_end_time_minutes_sin_next_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx+1, 'shift_end_time_minutes_sin']
                        user_mapped_sleep_shift_all_year_df.loc[idx, 'shift_end_time_minutes_cos_next_'+str(day)] = user_mapped_sleep_shift_all_year_df.loc[idx+1, 'shift_end_time_minutes_cos']

                        minutes_sleeping_features.add('off_day_next_'+str(day))
                        minutes_sleeping_features.add('minutes_of_duty_next_'+str(day))
                        minutes_sleeping_features.add('streak_next_'+str(day))
                        minutes_sleeping_features.add('shift_start_time_minutes_sin_next_'+str(day))
                        minutes_sleeping_features.add('shift_start_time_minutes_cos_next_'+str(day))
                        minutes_sleeping_features.add('shift_end_time_minutes_sin_next_'+str(day))
                        minutes_sleeping_features.add('shift_end_time_minutes_cos_next_'+str(day))

                        sleep_start_time_features.add('off_day_next_'+str(day))
                        sleep_start_time_features.add('minutes_of_duty_next_'+str(day))
                        sleep_start_time_features.add('streak_next_'+str(day))
                        sleep_start_time_features.add('shift_start_time_minutes_sin_next_'+str(day))
                        sleep_start_time_features.add('shift_start_time_minutes_cos_next_'+str(day))
                        sleep_start_time_features.add('shift_end_time_minutes_sin_next_'+str(day))
                        sleep_start_time_features.add('shift_end_time_minutes_cos_next_'+str(day))

        # user_mapped_sleep_shift_all_year_df = drop_null_values(user_mapped_sleep_shift_all_year_df).copy()
        FEATURES = USER_PROFILE_FEATURES + MODEL_FIXED_FEATURES
        for feature in FEATURES:
            minutes_sleeping_features.add(feature)
            sleep_start_time_features.add(feature)

            minutes_sleeping_features.add('minutes_sleeping')
            sleep_start_time_features.add('sleep_start_time_minutes_sin')

        minutes_sleeping_user_mapped_sleep_shift_all_year_shift_instance_df = user_mapped_sleep_shift_all_year_df[list(minutes_sleeping_features)]
        minutes_sleeping_user_mapped_sleep_shift_all_year_shift_instance_df = minutes_sleeping_user_mapped_sleep_shift_all_year_shift_instance_df.dropna()
        minutes_sleeping_user_mapped_sleep_shift_all_year_shift_instance_df.reset_index(inplace=True, drop=True)

        sleep_start_time_user_mapped_sleep_shift_all_year_shift_instance_df = user_mapped_sleep_shift_all_year_df[list(sleep_start_time_features)]
        sleep_start_time_user_mapped_sleep_shift_all_year_shift_instance_df = sleep_start_time_user_mapped_sleep_shift_all_year_shift_instance_df.dropna()
        sleep_start_time_user_mapped_sleep_shift_all_year_shift_instance_df.reset_index(inplace=True, drop=True)

        return minutes_sleeping_user_mapped_sleep_shift_all_year_shift_instance_df, sleep_start_time_user_mapped_sleep_shift_all_year_shift_instance_df


def main():
    """Main function of the program. This program preprocesses the user data. 
    """
    data_extraction = DataExtraction()
    data_preprocessing = DataPreprocessing()
    input = data_extraction.read_input()
    user_ids = data_extraction.extract_user_id_from_input(input)

    for user_id in user_ids:
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
        print(f'Completed data preprocessing for user with user id {user_id}')


if __name__ == "__main__":
    main()