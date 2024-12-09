from common.config_constants import INPUT_USER_FILE, USER_FEATURES_2020_DATASET, USER_FEATURES_2021_DATASET, USER_FEATURES_2022_DATASET, USER_FEATURES_2023_DATASET, \
                                    SLEEP_PROFILE_2020_DATASET, SLEEP_PROFILE_2021_DATASET, SLEEP_PROFILE_2022_DATASET, SLEEP_PROFILE_2023_DATASET, \
                                    SLEEP_PERIODS_2020_DATASET, SLEEP_PERIODS_2021_DATASET, SLEEP_PERIODS_2022_DATASET, SLEEP_PERIODS_2023_DATASET, \
                                    SLEEP_DAILY_2020_DATASET, SLEEP_DAILY_2021_DATASET, SLEEP_DAILY_2022_DATASET, SLEEP_DAILY_2023_DATASET, \
                                    SHIFT_INSTANCES_2020_DATASET, SHIFT_INSTANCES_2021_DATASET, SHIFT_INSTANCES_2022_DATASET, SHIFT_INSTANCES_2023_DATASET, \
                                    READI_SCORE_2020_DATASET, READI_SCORE_2021_DATASET, READI_SCORE_2022_DATASET, READI_SCORE_2023_DATASET
from common.functions import read_text_file, extract_user_id, read_csv_file, extract_user_data_from_combined_data, merge_data, remove_duplicate_values

class DataExtraction:

    def read_input(self):
        """This function is used to read input

        Returns:
            input_data (list): data stored as list (each element for each line) 
        """
        input_data = read_text_file(INPUT_USER_FILE)
        return input_data

    
    def extract_user_id_from_input(self, input_data):
        """This function extract's user id from the input data

        Args:
            input_data (list): list containing user id

        Returns:
            user_id (str): User ID of the user
        """
        user_ids = extract_user_id(input_data)
        return user_ids
    
    def read_data(self):
        """Read the csv files

        Returns:
            *dataframes (dataframe): Dataset to respective dataframes.
        """

        sleep_daily_2022_df = read_csv_file(SLEEP_DAILY_2022_DATASET)
        sleep_daily_2023_df = read_csv_file(SLEEP_DAILY_2023_DATASET)

        shift_instance_2022_df = read_csv_file(SHIFT_INSTANCES_2022_DATASET)
        shift_instance_2023_df = read_csv_file(SHIFT_INSTANCES_2023_DATASET)

        readi_score_2022_df = read_csv_file(READI_SCORE_2022_DATASET)
        readi_score_2023_df = read_csv_file(READI_SCORE_2023_DATASET)

        return sleep_daily_2022_df, sleep_daily_2023_df, shift_instance_2022_df, shift_instance_2023_df, \
               readi_score_2022_df, readi_score_2023_df

    
    def extract_user_data(self, sleep_daily_2022_df, sleep_daily_2023_df, \
                                shift_instance_2022_df, shift_instance_2023_df, \
                                readi_score_2022_df, readi_score_2023_df, user_id):
        """This function extract's and saves the user data

        Args:
            user_id (str): User ID of user who's data will be extracted
        """

        #Extract the user data from the dataset
        user_sleep_daily_2022_df = extract_user_data_from_combined_data(sleep_daily_2022_df, user_id)
        user_sleep_daily_2023_df = extract_user_data_from_combined_data(sleep_daily_2023_df, user_id)

        user_shift_instance_2022_df = extract_user_data_from_combined_data(shift_instance_2022_df, user_id)
        user_shift_instance_2023_df = extract_user_data_from_combined_data(shift_instance_2023_df, user_id)

        user_readi_score_2022_df = extract_user_data_from_combined_data(readi_score_2022_df, user_id)
        user_readi_score_2023_df = extract_user_data_from_combined_data(readi_score_2023_df, user_id)

        #Combine similar data for all years and remove the duplicates
        user_sleep_daily_df = merge_data(user_sleep_daily_2022_df, user_sleep_daily_2023_df)
        user_shift_instance_df = merge_data( user_shift_instance_2022_df, user_shift_instance_2023_df)
        user_readi_score_df = merge_data( user_readi_score_2022_df, user_readi_score_2023_df)

        user_sleep_daily_df = remove_duplicate_values(user_sleep_daily_df)
        user_shift_instance_df = remove_duplicate_values(user_shift_instance_df)
        user_readi_score_df = remove_duplicate_values(user_readi_score_df)

        #return the extracted data
        return user_sleep_daily_df, user_shift_instance_df, user_readi_score_df

def main():
    """Main function of the program. This program extracts user data from the dataset dump. 
    """
    data_extraction = DataExtraction()
    input = data_extraction.read_input()
    user_ids = data_extraction.extract_user_id_from_input(input)
    sleep_daily_2022_df, sleep_daily_2023_df, \
    shift_instance_2022_df, shift_instance_2023_df, \
    readi_score_2022_df, readi_score_2023_df = data_extraction.read_data()
    for user in user_ids:
        user_sleep_daily_df, user_shift_instance_df, user_readi_score_df = data_extraction.extract_user_data(sleep_daily_2022_df, sleep_daily_2023_df, \
                                                            shift_instance_2022_df, shift_instance_2023_df, readi_score_2022_df, readi_score_2023_df, user)
        print(f'Completed data extraction for user with user id {user}')

if __name__ == "__main__":
    main()