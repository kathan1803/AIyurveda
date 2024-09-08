import os
import random
import shutil
import numpy as np

def copy_random_images(src_dir, dst_dir, percentage=0.3):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        # Calculate the relative path and corresponding destination directory
        rel_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dst_dir, rel_path)

        # Filter image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # Determine the number of images to copy (30% of total images)
        num_images_to_copy = int(len(image_files) * percentage)

        # Select a random sample of images
        selected_images = np.random.choice(image_files, num_images_to_copy, replace=False)

        # Create destination directory if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        # Copy selected images to the destination directory
        for image in selected_images:
            shutil.copy(os.path.join(root, image), os.path.join(dest_path, image))

    print("Image selection and copying complete.")

# Example usage:
source_directory = 'test'
destination_directory = 'test_30'
copy_random_images(source_directory, destination_directory, percentage=0.3)