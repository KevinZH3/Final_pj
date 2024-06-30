import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from moviepy.editor import VideoClip

print("Starting the script...")

# 图片路径
image_paths = [
    '/nvme/wukai/3D-construction/1.jpg',
    '/nvme/wukai/3D-construction/2.jpg',
    '/nvme/wukai/3D-construction/3.jpg',
    '/nvme/wukai/3D-construction/4.jpg',
    '/nvme/wukai/3D-construction/5.jpg',
    '/nvme/wukai/3D-construction/6.jpg'
]

print("Loading and processing images...")

# 加载图片并转换为灰度图像
images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) for path in image_paths]

# 调整所有图像到相同大小
target_size = (images[0].shape[1], images[0].shape[0])
resized_images = [cv2.resize(img, target_size) for img in images]

print("Images loaded and resized.")

# 创建一个简单的3D网格
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 生成简单的网格
h, w = resized_images[0].shape
x = np.linspace(-1, 1, w)
y = np.linspace(-1, 1, h)
x, y = np.meshgrid(x, y)
z = np.mean(np.array(resized_images), axis=0) / 255.0  # 平均图片高度

print("Creating 3D plot...")

# 使用平均图片高度绘制表面
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='gray', edgecolor='none')

# 保存绘图
plt.savefig('3d_reconstruction_demo.png')

print("3D plot saved as 3d_reconstruction_demo.png")

# 定义更新视角的函数，用于视频
def make_frame(t):
    ax.view_init(30, t * 360 / 10)  # 在10秒内旋转360度
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

print("Creating video...")

# 创建视频
animation = VideoClip(make_frame, duration=10)
animation.write_videofile('3d_reconstruction_demo.mp4', fps=24)

print("Video saved as 3d_reconstruction_demo.mp4")

# 显示保存的3D重建图像
plt.show()

print("Script completed.")
