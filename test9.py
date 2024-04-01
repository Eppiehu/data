import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 创建一个画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制层
ax.add_patch(Rectangle((0.1, 0.6), 0.2, 0.1, edgecolor='black', facecolor='skyblue'))
ax.add_patch(Rectangle((0.4, 0.6), 0.2, 0.1, edgecolor='black', facecolor='skyblue'))
ax.add_patch(Rectangle((0.7, 0.6), 0.2, 0.1, edgecolor='black', facecolor='skyblue'))
ax.add_patch(Rectangle((0.1, 0.4), 0.2, 0.1, edgecolor='black', facecolor='skyblue'))
ax.add_patch(Rectangle((0.4, 0.4), 0.2, 0.1, edgecolor='black', facecolor='skyblue'))
ax.add_patch(Rectangle((0.7, 0.4), 0.2, 0.1, edgecolor='black', facecolor='skyblue'))
ax.add_patch(Rectangle((0.4, 0.2), 0.2, 0.1, edgecolor='black', facecolor='skyblue'))

# 标签
ax.text(0.2, 0.65, 'Conv+ReLU\n(64, 2x2)', ha='center', va='center')
ax.text(0.5, 0.65, 'Conv+ReLU\n(64, 2x2)', ha='center', va='center')
ax.text(0.8, 0.65, 'Flatten', ha='center', va='center')
ax.text(0.2, 0.45, 'Dense+ReLU\n(100)', ha='center', va='center')
ax.text(0.5, 0.45, 'Dense\n(2)', ha='center', va='center')
ax.text(0.8, 0.45, 'Compile\nAdam, MSE', ha='center', va='center')
ax.text(0.5, 0.25, 'Model Training', ha='center', va='center')

# 隐藏坐标轴
ax.axis('off')

# 显示图像
plt.show()
