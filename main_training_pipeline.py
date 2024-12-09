from main_extraction_pipeline import DataExtraction
from pandas import merge
from main_preprocessing_pipeline import DataPreprocessing
from main_modelling_pipeline import Modelling
from common.config_constants import SHIFT_INSTANCE
from common.functions import read_csv_file
from common.config_constants import USER_PROFILE_FEATURES


class Training:
    def training(self):
        """Training pipeline logic
        """
        data_extraction = DataExtraction()
        data_preprocessing = DataPreprocessing()
        input = data_extraction.read_input()
        user_ids = data_extraction.extract_user_id_from_input(input)
        modelling = Modelling()

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

            if user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df.shape[0] > 0 and user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df.shape[0] > 0:
                modelling.sleep_minutes_model(user_mapped_sleep_shift_all_year_day_sleep_minutes_modelling_df, 'minutes_sleeping', user_id, SHIFT_INSTANCE[0])
                modelling.sleep_start_time_model(user_mapped_sleep_shift_all_year_day_sleep_start_time_modelling_df, 'sleep_start_time_minutes_sin', user_id, SHIFT_INSTANCE[0])
            else:
                print('Data for Night Shift modelling not available')
            if user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df.shape[0] > 0 and user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df.shape[0] > 0:
                modelling.sleep_minutes_model(user_mapped_sleep_shift_all_year_night_sleep_minutes_modelling_df, 'minutes_sleeping', user_id, SHIFT_INSTANCE[1])
                modelling.sleep_start_time_model(user_mapped_sleep_shift_all_year_night_sleep_start_time_modelling_df, 'sleep_start_time_minutes_sin', user_id, SHIFT_INSTANCE[1])
            else:
                print('Data for Night Shift modelling not available')
            
            
            print(f'Completed model training for user with user id {user_id}')

def main():
    """Main function of the program. This program calls the training pipeline.
    """
    training = Training()
    training.training()

if __name__ == "__main__":
    main()