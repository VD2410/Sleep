from common.config_constants import LEARNING_RATE_DAY_SHIFT_SLEEP_MINUTES_MODEL, EPOCHS_DAY_SHIFT_SLEEP_MINUTES_MODEL, BATCH_SIZE_DAY_SHIFT_SLEEP_MINUTES_MODEL, \
                                    LEARNING_RATE_NIGHT_SHIFT_SLEEP_MINUTES_MODEL, EPOCHS_NIGHT_SHIFT_SLEEP_MINUTES_MODEL, BATCH_SIZE_NIGHT_SHIFT_SLEEP_MINUTES_MODEL, \
                                    LEARNING_RATE_DAY_SHIFT_SLEEP_START_TIME_MODEL, EPOCHS_DAY_SHIFT_SLEEP_START_TIME_MODEL, BATCH_SIZE_DAY_SHIFT_SLEEP_START_TIME_MODEL, \
                                    LEARNING_RATE_NIGHT_SHIFT_SLEEP_START_TIME_MODEL, EPOCHS_NIGHT_SHIFT_SLEEP_START_TIME_MODEL, BATCH_SIZE_NIGHT_SHIFT_SLEEP_START_TIME_MODEL, \
                                    PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS, CHECK_MODEL_ACCURACY
from common.functions import drop_column, min_max_scalar, train_test_split, prepare_dataset, save_model, load_model, bound_predicted_sin_values, \
                             convert_minutes_from_sin_values, adjust_sleep_time_for_night_shift, save_model_features, load_scalar_model
from architecture.model_architecture import day_shift_sleep_minutes_model_architecture, night_shift_sleep_minutes_model_architecture, \
                                            day_shift_sleep_start_time_model_architecture, night_shift_sleep_start_time_model_architecture

import pandas as pd
import tensorflow as tf
from keras import losses

class Modelling:
    
    def sleep_minutes_model(self, sleep_minutes_df, target_feature, user_id, shift_instance):
        """Sleep minutes model training and testing

        Args:
            sleep_minutes_df (dataframe): Dataframe with features and label for sleep minutes model
            user_id (str): User ID of user who's model will be trained
            shift_instance (str): obtain shift instance (day/night)
        """
        test_size = max(1, sleep_minutes_df.shape[0] - 30)
        sleep_minutes_df = drop_column(sleep_minutes_df, 'shift_date')
        features_without_target = [col for col in sleep_minutes_df.columns if col != target_feature]
        features = features_without_target.copy()
        features.append(target_feature)
        sleep_minutes_df = sleep_minutes_df[features]
        normalized_sleep_minutes_df = min_max_scalar(sleep_minutes_df, 'sleep_minutes', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(normalized_sleep_minutes_df, 'minutes_sleeping', test_size)
        save_model_features('sleep_minutes', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS, str(list(X_train.columns)))
        X_train_df, X_val_df, X_test_df, y_train_series, y_val_series, y_test_series = train_test_split(normalized_sleep_minutes_df, target_feature, test_size)
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(X_train_df), prepare_dataset(X_val_df), prepare_dataset(X_test_df), prepare_dataset(y_train_series), \
                                                     prepare_dataset(y_val_series), prepare_dataset(y_test_series)
        input_shape= (X_train.shape[1]) 
        if shift_instance == 'day':
            model = day_shift_sleep_minutes_model_architecture(input_shape)
            learning_rate = LEARNING_RATE_DAY_SHIFT_SLEEP_MINUTES_MODEL
        elif shift_instance == 'night':
            model = night_shift_sleep_minutes_model_architecture(input_shape)
            learning_rate = LEARNING_RATE_NIGHT_SHIFT_SLEEP_MINUTES_MODEL
        
        optim = tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate)
        msle = losses.Huber()
        model.compile(loss=msle, optimizer=optim)
        checkpoint = save_model('sleep_minutes', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS, monitor= 'val_loss', save_best_only=True, mode='min', verbose=0)
        
        if shift_instance == 'day':
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS_DAY_SHIFT_SLEEP_MINUTES_MODEL, batch_size=BATCH_SIZE_DAY_SHIFT_SLEEP_MINUTES_MODEL, callbacks=[checkpoint])
        elif shift_instance == 'night':
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS_NIGHT_SHIFT_SLEEP_MINUTES_MODEL, batch_size=BATCH_SIZE_NIGHT_SHIFT_SLEEP_MINUTES_MODEL, callbacks=[checkpoint])

        if CHECK_MODEL_ACCURACY:
            model = load_model('sleep_minutes', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        y_pred_train_normalized_data = model.predict(X_train)
        y_pred_test_normalized_data = model.predict(X_test)

        y_pred_train_normalized_data = pd.Series([y_pred_train_normalized_data[idx][0] for idx in range(len(y_pred_train_normalized_data))], name=target_feature)
        y_pred_test_normalized_data = pd.Series([y_pred_test_normalized_data[idx][0] for idx in range(len(y_pred_test_normalized_data))], name=target_feature)

        y_train = pd.Series(y_train, name=target_feature)
        y_test = pd.Series(y_test, name=target_feature)
        scalar_model = load_scalar_model('sleep_minutes', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)

        train_df = pd.concat([X_train_df, y_train], axis=1)
        train_df = scalar_model.inverse_transform(train_df)
        train_df = pd.DataFrame(train_df, columns=features)
        train_pred_df = pd.concat([X_train_df, y_pred_train_normalized_data], axis=1)
        train_pred_df = scalar_model.inverse_transform(train_pred_df)
        train_pred_df = pd.DataFrame(train_pred_df, columns=features)

        test_df = pd.concat([X_test_df, y_test], axis=1)
        test_df = scalar_model.inverse_transform(test_df)
        test_df = pd.DataFrame(test_df, columns=features)
        test_pred_df = pd.concat([X_test_df, y_pred_test_normalized_data], axis=1)
        test_pred_df = scalar_model.inverse_transform(test_pred_df)
        test_pred_df = pd.DataFrame(test_pred_df, columns=features)
        
        print(f'Train error: {abs(train_pred_df[target_feature] - train_df[target_feature]).mean()}')
        print(f'Test error: {abs(test_df[target_feature] - test_pred_df[target_feature]).mean()}')

    def sleep_start_time_model(self, sleep_start_time_df, target_feature, user_id, shift_instance):
        """Sleep start time model training and testing

        Args:
            sleep_start_time_df (dataframe): Dataframe with features and label for sleep start time model
            user_id (str): User ID of user who's model will be trained
            shift_instance (str): obtain shift instance (day/night)
        """    
        test_size = max(1, sleep_start_time_df.shape[0] - 30)
        sleep_start_time_df = drop_column(sleep_start_time_df, 'shift_date')
        features_without_target = [col for col in sleep_start_time_df.columns if col != target_feature]
        features = features_without_target.copy()
        features.append(target_feature)
        sleep_start_time_df = sleep_start_time_df[features]
        normalized_sleep_start_time_df = min_max_scalar(sleep_start_time_df, 'sleep_start_time', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(normalized_sleep_start_time_df, 'sleep_start_time_minutes_sin', test_size)
        save_model_features('sleep_start_time', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS, str(list(X_train.columns)))
        X_train_df, X_val_df, X_test_df, y_train_series, y_val_series, y_test_series = train_test_split(normalized_sleep_start_time_df, target_feature, test_size)
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(X_train_df), prepare_dataset(X_val_df), prepare_dataset(X_test_df), prepare_dataset(y_train_series), \
                                                     prepare_dataset(y_val_series), prepare_dataset(y_test_series)
        
        input_shape = (X_train.shape[1])

        if shift_instance == 'day':
            model = day_shift_sleep_start_time_model_architecture(input_shape)
            learning_rate = LEARNING_RATE_DAY_SHIFT_SLEEP_START_TIME_MODEL
        elif shift_instance == 'night':
            model = night_shift_sleep_start_time_model_architecture(input_shape)
            learning_rate = LEARNING_RATE_NIGHT_SHIFT_SLEEP_START_TIME_MODEL

        optim = tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate)
        msle = losses.Huber()
        model.compile(loss=msle, optimizer=optim)
        checkpoint = save_model('sleep_start_time', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS, monitor= 'val_loss', save_best_only=True, mode='min', verbose=0)

        if shift_instance == 'day':
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS_DAY_SHIFT_SLEEP_START_TIME_MODEL, batch_size=BATCH_SIZE_DAY_SHIFT_SLEEP_START_TIME_MODEL, callbacks=[checkpoint])
        elif shift_instance == 'night':
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS_NIGHT_SHIFT_SLEEP_START_TIME_MODEL, batch_size=BATCH_SIZE_NIGHT_SHIFT_SLEEP_START_TIME_MODEL, callbacks=[checkpoint])

        if CHECK_MODEL_ACCURACY:
            model = load_model('sleep_start_time', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
            y_pred_train_normalized_data = model.predict(X_train)
            y_pred_test_normalized_data = model.predict(X_test)

            y_pred_train_normalized_data = pd.Series([y_pred_train_normalized_data[idx][0] for idx in range(len(y_pred_train_normalized_data))])
            y_pred_test_normalized_data = pd.Series([y_pred_test_normalized_data[idx][0] for idx in range(len(y_pred_test_normalized_data))])

            y_train = pd.Series(y_train, name=target_feature)
            y_test = pd.Series(y_test, name=target_feature)
            scalar_model = load_scalar_model('sleep_start_time', user_id, shift_instance, PREVIOUS_DELTA_DAYS, NEXT_DELTA_DAYS)
            
            test_df = pd.concat([X_test_df, y_test], axis=1)
            test_df = scalar_model.inverse_transform(test_df)
            test_df = pd.DataFrame(test_df, columns=features)
            test_pred_df = pd.concat([X_test_df, y_pred_test_normalized_data], axis=1)
            test_pred_df = scalar_model.inverse_transform(test_pred_df)
            test_pred_df = pd.DataFrame(test_pred_df, columns=features)

            train_df = pd.concat([X_train_df, y_train], axis=1)
            train_df = scalar_model.inverse_transform(train_df)
            train_df = pd.DataFrame(train_df, columns=features)
            train_pred_df = pd.concat([X_train_df, y_pred_train_normalized_data], axis=1)
            train_pred_df = scalar_model.inverse_transform(train_pred_df)
            train_pred_df = pd.DataFrame(train_pred_df, columns=features)

            test_df[target_feature] = bound_predicted_sin_values(test_df[target_feature])
            test_pred_df[target_feature] = bound_predicted_sin_values(test_pred_df[target_feature])
            train_df[target_feature] = bound_predicted_sin_values(train_df[target_feature])
            train_pred_df[target_feature] = bound_predicted_sin_values(train_pred_df[target_feature])

            test_df[target_feature] = convert_minutes_from_sin_values(test_df[target_feature])
            test_pred_df[target_feature] = convert_minutes_from_sin_values(test_pred_df[target_feature])
            train_df[target_feature] = convert_minutes_from_sin_values(train_df[target_feature])
            train_pred_df[target_feature] = convert_minutes_from_sin_values(train_pred_df[target_feature])

            if shift_instance == 'night':
                test_df[target_feature] = adjust_sleep_time_for_night_shift(test_df[target_feature])
                test_pred_df[target_feature] = adjust_sleep_time_for_night_shift(test_pred_df[target_feature])

                train_df[target_feature] = adjust_sleep_time_for_night_shift(train_df[target_feature])
                train_pred_df[target_feature] = adjust_sleep_time_for_night_shift(train_pred_df[target_feature])

            
        print(f'Train error: {abs(train_pred_df[target_feature] - train_df[target_feature]).mean()}')
        print(f'Test error: {abs(test_df[target_feature] - test_pred_df[target_feature]).mean()}')