#!/bin/bash

# 定义图片路径和输出路径
IMAGE_PATH=processed_images
DATABASE_PATH=database.db
SPARSE_PATH=sparse
DENSE_PATH=dense

# 提取特征
colmap feature_extractor --database_path $DATABASE_PATH --image_path $IMAGE_PATH

# 匹配特征
colmap exhaustive_matcher --database_path $DATABASE_PATH

# 进行稀疏重建
mkdir $SPARSE_PATH
colmap mapper --database_path $DATABASE_PATH --image_path $IMAGE_PATH --output_path $SPARSE_PATH

# 进行密集重建
mkdir $DENSE_PATH
colmap image_undistorter --image_path $IMAGE_PATH --input_path $SPARSE_PATH/0 --output_path $DENSE_PATH --output_type COLMAP --max_image_size 2000
colmap patch_match_stereo --workspace_path $DENSE_PATH --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path $DENSE_PATH --workspace_format COLMAP --input_type geometric --output_path $DENSE_PATH/fused.ply
colmap poisson_mesher --input_path $DENSE_PATH/fused.ply --output_path $DENSE_PATH/meshed-poisson.ply

echo "COLMAP处理完成。"
