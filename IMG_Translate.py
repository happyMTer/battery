import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# 读取彩色图像
img = mpimg.imread('./result/model_structure/basic_2版_00.png')

# 将彩色图像转换为灰度图像
gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# 显示彩色图像和灰度图像
plt.figure(figsize=(10, 5))


plt.imshow(gray_img, cmap='gray')
plt.axis('off')

# 显示图形
plt.tight_layout()
plt.show()
plt.imsave('./result/model_structure/basic_2版_00_gray.png', gray_img, cmap='gray')
