from keras import layers, models

def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        layers.Conv1D(64, kernel_size=2, activation='relu'),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(2)  # 输出：偏移量和出射角度
    ])
    return model

# 假设的输入形状，您需要根据您的数据来设置这个值
example_input_shape = (10, 1)  # 例如：序列长度为10，特征数为1

# 创建一个模型实例来计算参数总数
model_example = build_cnn_model(example_input_shape)
total_params = model_example.count_params()
print("总参数数量:", total_params)
