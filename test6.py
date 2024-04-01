import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from model import build_cnn_model

def plot_model_parameters_with_layer_annotations(model_path):
    model = build_cnn_model((10, 1))  # 使用与训练时相同的input_shape
    model.load_weights(model_path)

    params = []
    layer_indices = []  # 用于存储每一层参数的开始索引
    current_index = 0

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:  # 检查层是否有权重
            layer_weights, layer_biases = weights
            params.extend(layer_weights.flatten().tolist())
            params.extend(layer_biases.flatten().tolist())
            layer_indices.append(current_index)
            current_index += len(layer_weights.flatten()) + len(layer_biases.flatten())

    plt.figure(figsize=(10, 6))
    plt.plot(params)
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title('Model Parameters Visualization')

    # 添加每一层的大致范围标注
    for i, index in enumerate(layer_indices):
        if i < len(layer_indices) - 1:
            plt.axvline(x=index, color='grey', linestyle='--')
            middle = (index + layer_indices[i + 1]) / 2
            plt.text(middle, max(params), f"Layer {i+1}", horizontalalignment='center')

    plt.show()

# 调用函数绘制模型参数
plot_model_parameters_with_layer_annotations("standard.h5")
