from model import build_cnn_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.models import load_model

def find_top_changes_indices(model_before, model_name, top_n):
    model_after = build_cnn_model((10, 1))
    model_after.load_weights(f'{model_name}.h5')

    all_changes = []
    for w1, w2 in zip(model_before.weights, model_after.weights):
        change = np.abs(w2.numpy() - w1.numpy())
        all_changes.append(change.flatten())

    # 合并所有层的变化量，并找出变化最大的top_n个参数的索引
    combined_changes = np.concatenate(all_changes)
    top_indices = np.argsort(combined_changes)[-top_n:]
    return top_indices

model = load_model("standard.h5")


top_n = 100 # 例如：提取前100个最大变化的参数
thicks_top_changes = [find_top_changes_indices(model, f'thick{i+1}', top_n) for i in range(3)]
materials_top_changes = [find_top_changes_indices(model, f'materials{i+1}', top_n) for i in range(3)]

def plot_change_distribution(top_changes_group, title, total_params=60000, bins=40):
    # 将所有模型的变化数据合并
    all_indices = np.concatenate(top_changes_group)

    # 绘制直方图
    plt.figure(figsize=(15, 5))
    plt.hist(all_indices, bins=bins, range=(0, total_params), alpha=0.7)
    plt.xlabel('Parameter Index')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(np.arange(0, total_params, 7000))  # 每500个参数标记一次横坐标
    plt.show()

plot_change_distribution(thicks_top_changes, 'Distribution of Top Changes in thicks Models')
plot_change_distribution(materials_top_changes, 'Distribution of Top Changes in materials Models')

def plot_histogram_comparison(thicks_changes, materials_changes, total_params=60000, bins=40):
    # 合并每组变化数据
    thicks_indices = np.concatenate(thicks_changes)
    materials_indices = np.concatenate(materials_changes)

    # 创建对比直方图
    plt.figure(figsize=(15, 7))

    plt.hist(thicks_indices, bins=bins, range=(0, total_params), alpha=0.5, color='blue', label='thicks Change')

    plt.hist(materials_indices, bins=bins, range=(0, total_params), alpha=0.5, color='green', label='materials Change')

    plt.xlabel('Parameter Index')
    plt.ylabel('Frequency')
    plt.title('Comparison of Parameter Change Distributions')
    plt.xticks(np.arange(0, total_params, 7000))  # 每500个参数标记一次横坐标
    plt.legend()
    plt.show()

# 调用函数绘制对比图
plot_histogram_comparison(thicks_top_changes, materials_top_changes)