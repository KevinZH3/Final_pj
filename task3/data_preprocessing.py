import os
import cv2

def load_images(image_folder):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            images.append(img)
    return images

def save_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(output_folder, f"image_{i:03d}.jpg"), img)

if __name__ == "__main__":
    input_folder = "path_to_your_images"
    output_folder = "processed_images"
    images = load_images(input_folder)
    save_images(images, output_folder)
    print(f"处理了 {len(images)} 张图片并保存到 {output_folder}")
