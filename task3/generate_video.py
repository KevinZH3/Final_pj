import open3d as o3d
import imageio

def generate_video(point_cloud_path, output_video_path):
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    ctr = vis.get_view_control()

    # 定义视频参数
    rot_angle = 10  # 每帧旋转角度
    n_frames = int(360 / rot_angle)

    # 捕获视频帧
    for i in range(n_frames):
        ctr.rotate(rot_angle, 0)
        vis.capture_screen_image(f"frame_{i:03d}.png")

    # 合成视频
    with imageio.get_writer(output_video_path, fps=24) as writer:
        for i in range(n_frames):
            image = imageio.imread(f"frame_{i:03d}.png")
            writer.append_data(image)

    print(f"视频保存为 {output_video_path}")

if __name__ == "__main__":
    point_cloud_path = "dense/meshed-poisson.ply"
    output_video_path = "3d_reconstruction_video.mp4"
    generate_video(point_cloud_path, output_video_path)
