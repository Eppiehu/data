import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义数据复制函数
def replicate_data(data, n_times):
    """将数据复制n次以增加维度"""
    replicated_data = np.repeat(data, n_times, axis=1)
    return replicated_data

# 数据准备
n1 = 1.0  # 空气的折射率
n2 = 1.33  # 水的折射率
thickness = 0.01  # 材料的厚度

incidence_angles = np.linspace(0, 90, 100)  # 生成0到90度之间的100个入射角度
incidence_angles_rad = np.radians(incidence_angles)  # 将角度转换为弧度

# 计算折射角和偏移量
refraction_angles_rad = np.arcsin(n1 / n2 * np.sin(incidence_angles_rad))
refraction_angles = np.degrees(refraction_angles_rad)  # 转换为角度
offsets = np.sin(refraction_angles_rad) * thickness  # 计算偏移量

# 组合成训练数据
X = incidence_angles.reshape(-1, 1)  # 输入数据：入射角度
y = np.vstack((offsets, refraction_angles)).T  # 输出数据：偏移量和出射角度

# 数据预处理：规范化
X_normalized = X / 90.0  # 假设入射角度的最大值为90度
y_normalized = y / np.max(y, axis=0)

# 将输入数据复制多次以增加维度
n_replications = 10  # 选择复制的次数
X_replicated = replicate_data(X_normalized, n_replications)

# 使用Seaborn的heatmap来可视化X_replicated
plt.figure(figsize=(12, 6))
sns.heatmap(X_replicated, cmap='viridis', cbar=True)
plt.title('Heatmap of X_replicated')
plt.xlabel('Replicated Dimension')
plt.ylabel('Sample Index')
plt.show()
