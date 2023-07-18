# This script deletes empty label files and their corresponding images from a specified folder.

import os

def delete_empty_labels_images(folder_path):
    images_folder = os.path.join(folder_path, 'images/train')
    labels_folder = os.path.join(folder_path, 'labels/train')

    labels_files = os.listdir(labels_folder)

    for label_file in labels_files:
        label_file_path = os.path.join(labels_folder, label_file)

        if os.stat(label_file_path).st_size == 0:
            os.remove(label_file_path)

            image_file_path = os.path.join(images_folder, label_file.replace('.txt', '.jpg'))
            os.remove(image_file_path)

            print(f"Deleted empty label file: {label_file} and its corresponding image.")

folder_path = '254473_import-images_001'  # Replace with the actual folder path

delete_empty_labels_images(folder_path)
