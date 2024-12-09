
import tensorflow as tf

def day_shift_sleep_minutes_model_architecture(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    dense_layer_1 = tf.keras.layers.Dense(units = 1024, activation='relu')(inputs)
    bat_norm = tf.keras.layers.BatchNormalization()(dense_layer_1)
    dense_layer_2 = tf.keras.layers.Dense(units = 512, activation='relu')(bat_norm)
    dense_layer_3 = tf.keras.layers.Dense(units = 256, activation='relu')(dense_layer_2)
    dense_layer_4 = tf.keras.layers.Dense(units = 128, activation='relu')(dense_layer_3)
    dense_layer_5 = tf.keras.layers.Dense(units = 64, activation='relu')(dense_layer_4)
    dense_layer_6 = tf.keras.layers.Dense(units = 32, activation='relu')(dense_layer_5)
    dense_layer_7 = tf.keras.layers.Dense(units = 16, activation='relu')(dense_layer_6)
    dense_layer_8 = tf.keras.layers.Dense(units = 8, activation='relu')(dense_layer_7)
    dense_layer_9 = tf.keras.layers.Dense(units = 4, activation='relu')(dense_layer_8)
    dense_layer_10 = tf.keras.layers.Dense(units=1)(dense_layer_9)
    model = tf.keras.Model(inputs=inputs, outputs=dense_layer_10)
    return model

def night_shift_sleep_minutes_model_architecture(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    dense_layer_1 = tf.keras.layers.Dense(units = 1024, activation='relu')(inputs)
    bat_norm = tf.keras.layers.BatchNormalization()(dense_layer_1)
    dense_layer_2 = tf.keras.layers.Dense(units = 512, activation='relu')(bat_norm)
    dense_layer_3 = tf.keras.layers.Dense(units = 256, activation='relu')(dense_layer_2)
    dense_layer_4 = tf.keras.layers.Dense(units = 128, activation='relu')(dense_layer_3)
    dense_layer_5 = tf.keras.layers.Dense(units = 64, activation='relu')(dense_layer_4)
    dense_layer_6 = tf.keras.layers.Dense(units = 32, activation='relu')(dense_layer_5)
    dense_layer_7 = tf.keras.layers.Dense(units = 16, activation='relu')(dense_layer_6)
    dense_layer_8 = tf.keras.layers.Dense(units = 8, activation='relu')(dense_layer_7)
    dense_layer_9 = tf.keras.layers.Dense(units = 4, activation='relu')(dense_layer_8)
    dense_layer_10 = tf.keras.layers.Dense(units=1)(dense_layer_9)
    model = tf.keras.Model(inputs=inputs, outputs=dense_layer_10)
    return model

def day_shift_sleep_start_time_model_architecture(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    dense_layer_1 = tf.keras.layers.Dense(units = 1024, activation='relu')(inputs)
    bat_norm = tf.keras.layers.BatchNormalization()(dense_layer_1)
    dense_layer_2 = tf.keras.layers.Dense(units = 512, activation='relu')(bat_norm)
    dense_layer_3 = tf.keras.layers.Dense(units = 256, activation='relu')(dense_layer_2)
    dense_layer_4 = tf.keras.layers.Dense(units = 128, activation='relu')(dense_layer_3)
    dense_layer_5 = tf.keras.layers.Dense(units = 64, activation='relu')(dense_layer_4)
    dense_layer_6 = tf.keras.layers.Dense(units = 32, activation='relu')(dense_layer_5)
    dense_layer_7 = tf.keras.layers.Dense(units = 16, activation='relu')(dense_layer_6)
    dense_layer_8 = tf.keras.layers.Dense(units = 8, activation='relu')(dense_layer_7)
    dense_layer_9 = tf.keras.layers.Dense(units = 4, activation='relu')(dense_layer_8)
    dense_layer_10 = tf.keras.layers.Dense(units=1)(dense_layer_9)
    model = tf.keras.Model(inputs=inputs, outputs=dense_layer_10)
    return model

def night_shift_sleep_start_time_model_architecture(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    dense_layer_1 = tf.keras.layers.Dense(units = 1024, activation='relu')(inputs)
    bat_norm = tf.keras.layers.BatchNormalization()(dense_layer_1)
    dense_layer_2 = tf.keras.layers.Dense(units = 512, activation='relu')(bat_norm)
    dense_layer_3 = tf.keras.layers.Dense(units = 256, activation='relu')(dense_layer_2)
    dense_layer_4 = tf.keras.layers.Dense(units = 128, activation='relu')(dense_layer_3)
    dense_layer_5 = tf.keras.layers.Dense(units = 64, activation='relu')(dense_layer_4)
    dense_layer_6 = tf.keras.layers.Dense(units = 32, activation='relu')(dense_layer_5)
    dense_layer_7 = tf.keras.layers.Dense(units = 16, activation='relu')(dense_layer_6)
    dense_layer_8 = tf.keras.layers.Dense(units = 8, activation='relu')(dense_layer_7)
    dense_layer_9 = tf.keras.layers.Dense(units = 4, activation='relu')(dense_layer_8)
    dense_layer_10 = tf.keras.layers.Dense(units=1)(dense_layer_9)
    model = tf.keras.Model(inputs=inputs, outputs=dense_layer_10)
    return model