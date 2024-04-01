import tensorflow as tf
from keras import layers, models

def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        layers.Conv1D(64, kernel_size=2, activation='relu'),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(2)  # 输出：偏移量和出射角度
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
