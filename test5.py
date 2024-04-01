import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

def load_model_and_weights(model_path):
    model = load_model(model_path)
    return model

def extract_layer_weights(model):
    # 提取模型中每层的权重
    return [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]

def compute_layerwise_correlation_matrix(base_model_layers, comparison_model_layers):
    # 计算层间的相关性矩阵
    num_layers = len(base_model_layers)
    correlation_matrix = np.zeros((num_layers, num_layers))
    for i, base_layer in enumerate(base_model_layers):
        for j, comp_layer in enumerate(comparison_model_layers):
            if base_layer.shape == comp_layer.shape:
                flattened_base = base_layer.flatten()
                flattened_comp = comp_layer.flatten()
                correlation = np.corrcoef(flattened_base, flattened_comp)[0, 1]
                correlation_matrix[i, j] = correlation
            else:
                correlation_matrix[i, j] = np.nan  # 标记为 NaN，表示无法比较
    return correlation_matrix

def plot_correlation_matrix(corr_matrix, title, layer_names):
    plt.figure(figsize=(8, 6))
#    ax = sns.heatmap(corr_matrix, annot=True, cmap='Blues')
    ax = sns.heatmap(corr_matrix, annot=True, cmap='Oranges', fmt='.4f',
                annot_kws={'size': 20, 'color': 'black'})
    cbar = ax.collections[0].colorbar  # 获取colorbar对象
    cbar.ax.tick_params(labelsize=12)  # 设置colorbar标签字体大小
    # 设置标题和轴标签
#    plt.title(title, fontsize=14)
#    plt.ylabel('Base Model Layer', fontsize=12)
#    plt.xlabel('Comparison Model Layer', fontsize=12)
#    plt.ylabel('', fontsize=12)
#    plt.xlabel('', fontsize=12)
    
    # 显示四周的边框，隐藏中间的网格线
#    ax.hlines([i for i in range(1, len(base_model_layers))], *ax.get_xlim(), color='white', linewidth=2)
#    ax.vlines([i for i in range(1, len(base_model_layers))], *ax.get_ylim(), color='white', linewidth=2)

    # 设置轴脊柱（边框）的可见性
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')

    # 设置坐标轴刻度字体大小
    ax.set_xticklabels(layer_names, fontsize=15, rotation=-45)  # 设置x轴标签和旋转角度
    ax.set_yticklabels(layer_names, fontsize=15, rotation=0)   # 设置y轴标签和旋转角度

    plt.subplots_adjust(bottom=0.15, top=0.90, left=0.15, right=0.90)

    
    plt.show()

# 加载基准模型并提取每层的权重
base_model = load_model_and_weights('standard.h5')
base_model_layers = extract_layer_weights(base_model)

# 定义材料和厚度模型的路径
materials_model_paths = [f'materials{i}.h5' for i in range(1, 4)]
thicks_model_paths = [f'thick{i}.h5' for i in range(1, 4)]

layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']

# 对每个材料模型和厚度模型绘制相关性矩阵图
for i, path in enumerate(materials_model_paths):
    comparison_model = load_model_and_weights(path)
    comparison_model_layers = extract_layer_weights(comparison_model)
    corr_matrix = compute_layerwise_correlation_matrix(base_model_layers, comparison_model_layers)
#    plot_correlation_matrix(corr_matrix, f'Materials Model {i+1} Layer-wise Correlation Matrix', layer_names)
    plot_correlation_matrix(corr_matrix, '', layer_names)

for i, path in enumerate(thicks_model_paths):
    comparison_model = load_model_and_weights(path)
    comparison_model_layers = extract_layer_weights(comparison_model)
    corr_matrix = compute_layerwise_correlation_matrix(base_model_layers, comparison_model_layers)
#    plot_correlation_matrix(corr_matrix, f'Thicks Model {i+1} Layer-wise Correlation Matrix', layer_names)
    plot_correlation_matrix(corr_matrix, '', layer_names)




def compute_euclidean_distance(base_layer, comparison_layer):
    flattened_base = base_layer.flatten()
    flattened_comp = comparison_layer.flatten()
    return np.linalg.norm(flattened_base - flattened_comp)

def plot_layerwise_distance(base_model_layers, comparison_model_layers, title):
    distances = [compute_euclidean_distance(base_layer, comp_layer) 
                 for base_layer, comp_layer in zip(base_model_layers, comparison_model_layers)]
    
    # 绘制每层的欧几里得距离
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(distances)), distances, color='blue')
    plt.xlabel('Layer Number')
    plt.ylabel('Euclidean Distance')
    plt.title(title)
    plt.show()

# 对每个模型绘制欧几里得距离图
for i, path in enumerate(materials_model_paths + thicks_model_paths):
    comparison_model = load_model_and_weights(path)
    comparison_model_layers = extract_layer_weights(comparison_model)
    plot_layerwise_distance(base_model_layers, comparison_model_layers, f'Model {i+1} Layer-wise Euclidean Distance')




def compute_euclidean_distances(base_model_layers, comparison_model_layers):
    distances = []
    for base_layer, comp_layer in zip(base_model_layers, comparison_model_layers):
        flattened_base = base_layer.flatten()
        flattened_comp = comp_layer.flatten()
        distance = np.linalg.norm(flattened_base - flattened_comp)
        distances.append(distance)
    return distances

# 收集所有模型与基准模型的层间距离
all_distances = []

# 材料模型
for path in materials_model_paths:
    comparison_model = load_model_and_weights(path)
    comparison_model_layers = extract_layer_weights(comparison_model)
    distances = compute_euclidean_distances(base_model_layers, comparison_model_layers)
    all_distances.append(distances)

# 厚度模型
for path in thicks_model_paths:
    comparison_model = load_model_and_weights(path)
    comparison_model_layers = extract_layer_weights(comparison_model)
    distances = compute_euclidean_distances(base_model_layers, comparison_model_layers)
    all_distances.append(distances)

plt.figure(figsize=(12, 6))
# 定义新的标签
labels = ['Materials-A', 'Materials-B', 'Materials-C', 'Thicks-D', 'Thicks-E', 'Thicks-F']

# 绘制箱线图并设置新的x轴标签
plt.figure(figsize=(12, 3))
box = plt.boxplot(all_distances, labels=labels)
#plt.xlabel('Model', fontsize=14)  # 设置横坐标轴标题和字体大小
plt.ylabel('Euclidean Distance', fontsize=14)  # 设置纵坐标轴标题和字体大小

# 设置坐标轴刻度标签的字体大小和旋转角度
plt.setp(box['whiskers'], color='black')  # 将盒须的颜色设置为黑色，如果需要
plt.tick_params(axis='x', labelsize=12, rotation=0)  # 横坐标轴刻度标签
plt.tick_params(axis='y', labelsize=12, rotation=0)  # 纵坐标轴刻度标签
plt.subplots_adjust(bottom=0.10, top=0.95, left=0.10, right=0.95)

plt.show()


