import matplotlib.pyplot as plt
import numpy as np

# 数据准备
n1 = 1.0  # 空气的折射率
n2 = 1.33  # 水的折射率
thickness = 0.01  # 材料的厚度

incidence_angles = np.linspace(0, 90, 100)  # 生成0到90度之间的100个入射角度
incidence_angles_rad = np.radians(incidence_angles)  # 将角度转换为弧度

# 计算折射角和偏移量
refraction_angles_rad = np.arcsin(n1 / n2 * np.sin(incidence_angles_rad))
refraction_angles = np.degrees(refraction_angles_rad)  # 转换为角度
offsets = thickness / np.cos(refraction_angles_rad)  # 计算偏移量

# 设置标注和刻度的字号
label_fontsize = 18
tick_fontsize = 16

# 创建绘图和第一个纵坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制入射角度与偏移量的关系（使用第一个纵坐标轴）
color = 'navy'
ax1.set_xlabel('Incidence Angle (degrees)', fontsize=20)
ax1.set_ylabel('Offset (mm)', color=color, fontsize=20)
ax1.plot(incidence_angles, offsets, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
ax1.tick_params(axis='x', labelsize=20)

# 调整x轴和y轴的刻度数量
ax1.locator_params(axis='x', nbins=6)
ax1.locator_params(axis='y', nbins=6)

# 创建第二个纵坐标轴，共享相同的x轴
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Refraction Angle (degrees)', color=color, fontsize=20)
ax2.plot(incidence_angles, refraction_angles, color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

# 显示图表
plt.show()
