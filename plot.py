# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
#
# # 设置字体以正确显示中文
# plt.rcParams['font.family'] = 'STFangsong'
#
# # 动态接收输入的图片路径
# image_paths = {
#     (0, 0): 'D:/05_Learning_file/_毕业设计/结果/载体.png',
#     (1, 0): 'D:/05_Learning_file/_毕业设计/结果/水印.png',
#     (0, 1): 'D:/05_Learning_file/_毕业设计/结果/3_加密.png',
#     (1, 1): 'D:/05_Learning_file/_毕业设计/结果/7_提取.png',
# }
#
# # 原始图片的标签
# labels = {
#     (0, 0): '载\n体\n图\n像',
#     (1, 0): '水\n印\n图\n像'
# }
#
# # 动态接收输入的标题
# titles = {
#     (0, 0): '载体图像',
#     (1, 0): '水印图像',
#     (0, 1): '含水印图像',
#     (1, 1): '提取出的水印图像',
#     (0, 2): '载体图像细节',
#     (1, 2): '水印图像细节',
#     (0, 3): '含水印图像细节',
#     (1, 3): '提取后的水印图像细节'
# }
#
# # 动态确定行数和列数
# nrows = max(key[0] for key in image_paths.keys()) + 1
# ncols = max(key[1] for key in image_paths.keys()) + 1
#
# # 初始化画布和子图数组
# fig, axes = plt.subplots(nrows=nrows, ncols=2 * ncols, figsize=(16, 8))
#
# # 如果axes是一维数组，将其转换为二维数组
# if nrows == 1:
#     axes = np.expand_dims(axes, axis=0)
# if ncols == 1:
#     axes = np.expand_dims(axes, axis=1)
#
# # 裁剪并显示函数
# def crop_and_display(image_path, ax, sx, sy, size=200, show_rectangle=False):
#     img = Image.open(image_path)
#     img_array = np.array(img)
#     cropped = img_array[sx:sx + size, sy:sy + size]
#     ax.imshow(cropped, cmap='gray', interpolation='nearest')
#     ax.axis('off')
#
# # 获取裁剪区域并记录每排图像的裁剪位置
# start_xs = [None] * nrows
# start_ys = [None] * nrows
#
# for i in range(nrows):
#     for j in range(ncols):
#         size = 200
#         if (i, j) in image_paths:
#             # 显示原图
#             img = Image.open(image_paths[(i, j)])
#             axes[i, j].imshow(img, cmap='gray')
#             # if (i, 0) in labels:
#                 # axes[i, 0].set_ylabel(labels.get((i, 0), ''), fontsize=20, rotation=0, labelpad=20, verticalalignment='center')
#             axes[i, j].set_xticks([])
#             axes[i, j].set_yticks([])
#
#             if start_xs[i] is None and start_ys[i] is None:
#                 max_x = img.width - size
#                 max_y = img.height - size
#                 start_x = np.random.randint(0, max_x)
#                 start_y = np.random.randint(0, max_y)
#                 start_xs[i] = start_x
#                 start_ys[i] = start_y
#             else:
#                 start_x = start_xs[i]
#                 start_y = start_ys[i]
#             rect = plt.Rectangle((start_y, start_x), size, size, linewidth=1, edgecolor='r', facecolor='none')
#             axes[i, j].add_patch(rect)
#
#             # 显示原图细节
#             crop_and_display(image_paths[(i, j)], axes[i, j + ncols], start_x, start_y)
#             if (i, j + ncols) in titles:
#                 axes[i, j + ncols].set_title(titles.get((i, j + ncols), ''), fontsize=20)
#
# # 设置每个子图的标题
# for i in range(nrows):
#     for j in range(ncols):
#         if (i, j) in titles:
#             axes[i, j].set_title(titles.get((i, j), ''), fontsize=20)
#
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.standard_normal(30).cumsum(), color='black', linestyle='-', marker='o')
plt.show()