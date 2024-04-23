import matplotlib.pyplot as plt
import numpy as np

 # 根据结果，进行画图
labels = ['YOLOv3', 'YOLOv5s', 'YOLOv8s']
# Recall
clean = np.array([100, 100, 100])
config1 = np.array([60.7, 64.9, 69.7])
config2 = np.array([23.2, 40.5, 51.5])
config3 = np.array([33.9, 62.2, 51.5])
config4 = np.array([1.8, 8.1, 27.3])
# Object
# clean = np.array([82, 58, 29])
# config1 = np.array([13537, 10833, 3970])
# config2 = np.array([16950, 13310, 5011])
# config3 = np.array([16835, 15026, 5497])
# config4 = np.array([21845, 19926, 6805])
# NMS Time
# clean = np.array([1, 1, 1])
# config1 = np.array([82, 58, 8])
# config2 = np.array([159, 104, 11])
# config3 = np.array([149, 114, 11])
# config4 = np.array([286, 237, 16])

x = np.arange(len(labels))
total_width, n = 0.8, 5
width = total_width / n

def operate(x, y):
    for i,j in zip(x, y):
        plt.text(i, j + 1, str(j), ha='center')
    return x + width

fig, ax = plt.subplots(figsize=(10, 3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.patch.set_color('gainsboro')
ax.bar(x, clean, width=width, label='clean', color='#8ECFC9')   #, label='clean'
x = operate(x, clean)

ax.bar(x, config1, width, label='config1', color='#FFBE7A')
x = operate(x, config1)

ax.bar(x, config2, width, label='config2', color='#FA7F6F')
x = operate(x, config2)

ax.bar(x, config3, width, label='config3', color='#82D0D2')
x = operate(x, config3)

ax.bar(x, config4, width, label='config4', color='#BEB8DC')
operate(x, config4)

fontsize = 18
ax.set_ylabel('Recall [%]', fontsize=fontsize)
# ax.set_xlabel('Model', fontsize=fontsize)
# ax.set_yticks([0, 50, 100])
# ax.set_xticks([])
ax.set_xticks(np.arange(len(labels)) + 0.3)
ax.set_xticklabels(labels, size=fontsize)

# ax.set_title('Metric=', fontsize=fontsize)
# ax.legend(fontsize=12)

#设置网格刻度
plt.grid(True,linestyle=':',color='gray',alpha=0.6)
# plt.axis('off')
# plt.legend(loc="upper left") # label的位置在左上，没有这句会找不到label去哪了
# dpi放大不模糊
plt.savefig('./modelxx.png', dpi=300)
# plt.savefig('./model.svg', format='svg')  # svg矢量图，生成的文件效果一样大小更小
# plt.show()

