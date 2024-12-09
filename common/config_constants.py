from common.utils import load_config_file

config = load_config_file('config/config.yaml')

config_directory_path = config['dir_path']
config_datasets = config['datasets']

CONFIG_PATH = config_directory_path['path_to_config_files']

DATASET_PATH = config_directory_path['path_to_datasets']
RAW_DATASETS_PATH = DATASET_PATH + config_directory_path['path_to_load_datasets']
GENERATED_DATASETS_PATH = DATASET_PATH + config_directory_path['path_to_save_datasets']

USER_FEATURES_DATASET_SUBPATH = config_directory_path['path_to_user_features_datasets']
SLEEP_PROFILE_DATASET_SUBPATH = config_directory_path['path_to_sleep_profile_datasets']
SLEEP_PERIODS_DATASET_SUBPATH = config_directory_path['path_to_sleep_periods_datasets']
SLEEP_DAILY_DATASET_SUBPATH = config_directory_path['path_to_sleep_daily_datasets']
SHIFT_INSTANCES_DATASET_SUBPATH = config_directory_path['path_to_shift_instances_datasets']
READI_SCORE_DATASET_SUBPATH = config_directory_path['path_to_readi_score_datasets']

USER_FEATURES_DATASET_PATH = RAW_DATASETS_PATH + USER_FEATURES_DATASET_SUBPATH
SLEEP_PROFILE_DATASET_PATH = RAW_DATASETS_PATH + SLEEP_PROFILE_DATASET_SUBPATH
SLEEP_PERIODS_DATASET_PATH = RAW_DATASETS_PATH + SLEEP_PERIODS_DATASET_SUBPATH
SLEEP_DAILY_DATASET_PATH = RAW_DATASETS_PATH + SLEEP_DAILY_DATASET_SUBPATH
SHIFT_INSTANCES_DATASET_PATH = RAW_DATASETS_PATH + SHIFT_INSTANCES_DATASET_SUBPATH
READI_SCORE_DATASET_PATH = RAW_DATASETS_PATH + READI_SCORE_DATASET_SUBPATH

USER_FEATURES_2020_DATASET = USER_FEATURES_DATASET_PATH + config_datasets['user_features_2020']
USER_FEATURES_2021_DATASET = USER_FEATURES_DATASET_PATH + config_datasets['user_features_2021']
USER_FEATURES_2022_DATASET = USER_FEATURES_DATASET_PATH + config_datasets['user_features_2022']
USER_FEATURES_2023_DATASET = USER_FEATURES_DATASET_PATH + config_datasets['user_features_2023']

SLEEP_PROFILE_2020_DATASET = SLEEP_PROFILE_DATASET_PATH + config_datasets['sleep_profiles_2020']
SLEEP_PROFILE_2021_DATASET = SLEEP_PROFILE_DATASET_PATH + config_datasets['sleep_profiles_2021']
SLEEP_PROFILE_2022_DATASET = SLEEP_PROFILE_DATASET_PATH + config_datasets['sleep_profiles_2022']
SLEEP_PROFILE_2023_DATASET = SLEEP_PROFILE_DATASET_PATH + config_datasets['sleep_profiles_2023']

SLEEP_PERIODS_2020_DATASET = SLEEP_PERIODS_DATASET_PATH + config_datasets['sleep_periods_2020']
SLEEP_PERIODS_2021_DATASET = SLEEP_PERIODS_DATASET_PATH + config_datasets['sleep_periods_2021']
SLEEP_PERIODS_2022_DATASET = SLEEP_PERIODS_DATASET_PATH + config_datasets['sleep_periods_2022']
SLEEP_PERIODS_2023_DATASET = SLEEP_PERIODS_DATASET_PATH + config_datasets['sleep_periods_2023']

SLEEP_DAILY_2020_DATASET = SLEEP_DAILY_DATASET_PATH + config_datasets['sleep_daily_2020']
SLEEP_DAILY_2021_DATASET = SLEEP_DAILY_DATASET_PATH + config_datasets['sleep_daily_2021']
SLEEP_DAILY_2022_DATASET = SLEEP_DAILY_DATASET_PATH + config_datasets['sleep_daily_2022']
SLEEP_DAILY_2023_DATASET = SLEEP_DAILY_DATASET_PATH + config_datasets['sleep_daily_2023']

SHIFT_INSTANCES_2020_DATASET = SHIFT_INSTANCES_DATASET_PATH + config_datasets['shift_instances_2020']
SHIFT_INSTANCES_2021_DATASET = SHIFT_INSTANCES_DATASET_PATH + config_datasets['shift_instances_2021']
SHIFT_INSTANCES_2022_DATASET = SHIFT_INSTANCES_DATASET_PATH + config_datasets['shift_instances_2022']
SHIFT_INSTANCES_2023_DATASET = SHIFT_INSTANCES_DATASET_PATH + config_datasets['shift_instances_2023']

READI_SCORE_2020_DATASET = READI_SCORE_DATASET_PATH + config_datasets['readi_score_2020']
READI_SCORE_2021_DATASET = READI_SCORE_DATASET_PATH + config_datasets['readi_score_2021']
READI_SCORE_2022_DATASET = READI_SCORE_DATASET_PATH + config_datasets['readi_score_2022']
READI_SCORE_2023_DATASET = READI_SCORE_DATASET_PATH + config_datasets['readi_score_2023']


user_input_config = load_config_file('config/input_config.yaml')
prediction_range_data = user_input_config['prediction_range']

PREDICTION_START_DATE = prediction_range_data['start_date']
PREDICTION_TOTAL_DAYS_TO_PREDICT = prediction_range_data['total_days_to_predict']

constant_config = load_config_file('config/constants.yaml')

feature_list = constant_config['features']
variable_list = constant_config['variables']
delta_days_data = constant_config['delta_days']
hyperparameter_list = constant_config['hyperparameters']
user_input_data = constant_config['user_list']

INPUT_USER_FILE = CONFIG_PATH + user_input_data['user_list_file_name']
USER_SHIFT_START_END_FEATURES = feature_list['user_shift_start_end_features']
USER_SLEEP_START_END_FEATURES = feature_list['user_sleep_start_end_features']
USER_PROFILE_FEATURES = feature_list['user_profile_features']
MODEL_FIXED_FEATURES = feature_list['model_fixed_features']
USER_SLEEP_SHIFT_TRANSFORMATION_FEATURES = feature_list['user_sleep_shift_features_for_transformation']
SHIFT_INSTANCE = variable_list['shift_instance']
CHECK_MODEL_ACCURACY = variable_list['check_model_accuracy']
PREVIOUS_DELTA_DAYS = delta_days_data['previous_delta_days']
NEXT_DELTA_DAYS = delta_days_data['next_delta_days']

hyperparameter_list = constant_config['hyperparameters']

day_shift_sleep_minutes_model = hyperparameter_list['day_shift_sleep_minutes_model']
night_shift_sleep_minutes_model = hyperparameter_list['night_shift_sleep_minutes_model']
day_shift_sleep_start_model = hyperparameter_list['day_shift_sleep_start_model']
night_shift_sleep_start_model = hyperparameter_list['night_shift_sleep_start_model']

LEARNING_RATE_DAY_SHIFT_SLEEP_MINUTES_MODEL = day_shift_sleep_minutes_model['learning_rate']
EPOCHS_DAY_SHIFT_SLEEP_MINUTES_MODEL = day_shift_sleep_minutes_model['epoch']
BATCH_SIZE_DAY_SHIFT_SLEEP_MINUTES_MODEL = day_shift_sleep_minutes_model['batch_size']

LEARNING_RATE_NIGHT_SHIFT_SLEEP_MINUTES_MODEL = night_shift_sleep_minutes_model['learning_rate']
EPOCHS_NIGHT_SHIFT_SLEEP_MINUTES_MODEL = night_shift_sleep_minutes_model['epoch']
BATCH_SIZE_NIGHT_SHIFT_SLEEP_MINUTES_MODEL = night_shift_sleep_minutes_model['batch_size']

LEARNING_RATE_DAY_SHIFT_SLEEP_START_TIME_MODEL = day_shift_sleep_start_model['learning_rate']
EPOCHS_DAY_SHIFT_SLEEP_START_TIME_MODEL = day_shift_sleep_start_model['epoch']
BATCH_SIZE_DAY_SHIFT_SLEEP_START_TIME_MODEL = day_shift_sleep_start_model['batch_size']

LEARNING_RATE_NIGHT_SHIFT_SLEEP_START_TIME_MODEL = night_shift_sleep_minutes_model['learning_rate']
EPOCHS_NIGHT_SHIFT_SLEEP_START_TIME_MODEL = night_shift_sleep_minutes_model['epoch']
BATCH_SIZE_NIGHT_SHIFT_SLEEP_START_TIME_MODEL = night_shift_sleep_minutes_model['batch_size']