import numpy as np
import tensorflow as tf
from keras import layers, models
from model import build_cnn_model
from keras.callbacks import EarlyStopping

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
X_cnn = np.expand_dims(X_replicated, axis=2)  # 添加一个额外的维度

# 定义和训练CNN模型
model = build_cnn_model((X_cnn.shape[1], X_cnn.shape[2]))

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    loss_weights=[1.0, 1.0]  # 假设第一个输出（偏移量）的重要性是第二个输出（出射角度）的两倍
)


# 创建EarlyStopping回调
early_stopping = EarlyStopping(
    monitor='loss',   # 监控模型的训练损失
    patience=30,      # 在10个轮次后若性能不再改善，则停止训练
    verbose=1,        # 打印消息
    mode='min'        # 因为我们监控的是损失，所以模式是'min'
)

# 训练模型，并加入回调
model.fit(X_cnn, y_normalized, epochs=1000, batch_size=10, callbacks=[early_stopping])

# 保存模型
model.save("standard.h5")
