# 基于NeRF和COLMAP的3D重建和新视图合成

## 概述

本项目演示了使用NeRF和COLMAP进行3D重建和新视图合成的过程。包括以下步骤：

1. 数据预处理
2. 使用COLMAP进行相机参数估计
3. NeRF的训练和推理
4. 生成3D重建视频

## 环境设置

### 先决条件

- Python 3.6+
- COLMAP
- Open3D
- Imageio
- NeRF

### 安装

1. 从[COLMAP GitHub](https://github.com/colmap/colmap)安装COLMAP
2. 安装Python库：
    ```sh
    pip install open3d imageio matplotlib
    ```

## 使用方法

### 1. 数据预处理

运行以下命令以预处理你的图片：
```sh
python data_preprocessing.py
 ```

### 2. 相机参数估计
运行以下脚本使用COLMAP进行相机参数估计：
```bash colmap_process.sh
```
### 3. NeRF训练和推理
运行以下命令训练NeRF模型：
```python nerf_train.py
```
### 4. 生成3D重建视频
运行以下命令生成3D重建视频：
```python generate_video.py
```
