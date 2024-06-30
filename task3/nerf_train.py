import os

def run_nerf_training(data_path, output_path):
    os.system(f"python run_nerf.py --config configs/llff_data/fern.txt --datadir {data_path} --basedir {output_path}")

if __name__ == "__main__":
    data_path = "dense"
    output_path = "nerf_output"
    run_nerf_training(data_path, output_path)
    print(f"NeRF训练完成。请检查 {output_path} 以获取结果。")
