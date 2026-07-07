import os
import random
import shutil

original_dataset_root = r'dataset'
new_dataset_root = r'reduced_data'
images_per_subfolder = 500

if not os.path.exists(new_dataset_root):
    os.makedirs(new_dataset_root)
    print(f"Created new dataset root: {new_dataset_root}")
else:
    print(f"New dataset root already exists: {new_dataset_root}")

subfolders = [d for d in os.listdir(original_dataset_root) if os.path.isdir(os.path.join(original_dataset_root, d))]
print(f"Found {len(subfolders)} subfolders in the original dataset.")

for subfolder_name in subfolders:
    original_subfolder_path = os.path.join(original_dataset_root, subfolder_name)
    new_subfolder_path = os.path.join(new_dataset_root, subfolder_name)

    if not os.path.exists(new_subfolder_path):
        os.makedirs(new_subfolder_path)
        print(f"Created new subfolder: {new_subfolder_path}")
    
    image_files = [f for f in os.listdir(original_subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    print(f" Found {len(image_files)} images in {subfolder_name}.")

    if len(image_files) < images_per_subfolder:
        print(f"Not enough images in '{subfolder_name}' ({len(image_files)}). Copying all available.")
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, images_per_subfolder)
        print(f"Selected {len(selected_images)} random images from '{subfolder_name}'")

    for image_file_name in selected_images:
        original_image_path = os.path.join(original_subfolder_path, image_file_name)
        new_image_path = os.path.join(new_subfolder_path, image_file_name)
        try:
            shutil.copy2(original_image_path, new_image_path)
        except Exception as e:
            print(f"Not successful copying {original_image_path} to {new_image_path}: {e}")

print("\nDataset reduction complete!")
print(f"New reduced dataset should be located at: {new_dataset_root}")
print(f"Each subfolder should now contain approximately {images_per_subfolder} images.")
