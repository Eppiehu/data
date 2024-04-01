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

# 绘制单个模型的相关性矩阵热图
def plot_individual_correlation_matrix(base_weights, model_weights, title):
    correlation_matrix = compute_correlation_matrix([base_weights, model_weights])
    plt.figure(figsize=(5, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
    plt.title(title)
    plt.show()

# 收集不同模型的相关性数据
def collect_correlation_data(base_weights, model_weights_list):
    correlations = []
    for weights in model_weights_list:
        corr_matrix = compute_correlation_matrix([base_weights, weights])
        correlations.append(corr_matrix[0, 1])  # 提取基准模型和当前模型之间的相关性
    return correlations

# 加载基准模型并提取权重
base_model = load_model_and_weights('standard.h5')
base_weights = extract_weights(base_model)

# 定义其他模型的路径
materials_model_paths = [f'materials{i}.h5' for i in range(1, 4)]
thicks_model_paths = [f'thick{i}.h5' for i in range(1, 4)]

# 加载其他模型并提取权重
materials_weights = [extract_weights(load_model_and_weights(path)) for path in materials_model_paths]
thicks_weights = [extract_weights(load_model_and_weights(path)) for path in thicks_model_paths]

# 为每个模型绘制相关性矩阵热图
for i, path in enumerate(materials_model_paths):
    model_weights = extract_weights(load_model_and_weights(path))
    plot_individual_correlation_matrix(base_weights, model_weights, f'Materials Model {i+1} Correlation Matrix')

for i, path in enumerate(thicks_model_paths):
    model_weights = extract_weights(load_model_and_weights(path))
    plot_individual_correlation_matrix(base_weights, model_weights, f'Thicks Model {i+1} Correlation Matrix')

# 收集相关性数据并绘制箱线图
materials_correlations = collect_correlation_data(base_weights, materials_weights)
thicks_correlations = collect_correlation_data(base_weights, thicks_weights)

plt.figure(figsize=(10, 6))
plt.boxplot([materials_correlations, thicks_correlations], labels=['Materials', 'Thicks'])
plt.title('Correlation Distribution Comparison')
plt.ylabel('Correlation')
plt.show()
