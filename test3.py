import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

def load_model_and_weights(model_path):
    model = load_model(model_path)
    return model

def extract_weights(model):
    weights = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    return flattened_weights

def compute_correlation_matrix(weights_list):
    return np.corrcoef(weights_list)

# 绘制热图
def plot_correlation_matrix(corr_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues')
    plt.title(title)
    plt.show()

# 加载基准模型并提取权重
base_model = load_model_and_weights('standard.h5')
base_weights = extract_weights(base_model)
# 修改模型路径列表
materials_model_paths = [f'materials{i}.h5' for i in range(1, 4)]
thicks_model_paths = [f'thick{i}.h5' for i in range(1, 4)]

# 加载模型并提取权重
materials_weights = [extract_weights(load_model_and_weights(path)) for path in materials_model_paths]
thicks_weights = [extract_weights(load_model_and_weights(path)) for path in thicks_model_paths]

# 计算相关性矩阵
materials_correlation = compute_correlation_matrix([base_weights] + materials_weights)
thicks_correlation = compute_correlation_matrix([base_weights] + thicks_weights)

# 绘制热图
plot_correlation_matrix(materials_correlation, 'Materials Model Weights Correlation Matrix')
plot_correlation_matrix(thicks_correlation, 'Thicks Model Weights Correlation Matrix')

