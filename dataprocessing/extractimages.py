import numpy as np
import pandas as pd
import os
from PIL import Image

dataset_path = 'breastmnist_224.npz'
data = np.load(dataset_path)

splits = {
    'train': (data['train_images'], data['train_labels']),
    'val': (data['val_images'], data['val_labels']),
    'test': (data['test_images'], data['test_labels']),
}

base_output_dir = 'BreastMNIST_Folder'
image_output_dir = os.path.join(base_output_dir, 'images')
os.makedirs(image_output_dir, exist_ok=True)

def save_image_array(image_array, output_path):
    image = Image.fromarray(image_array.astype('uint8'))
    image.save(output_path)

all_dataframes = []

for split, (images, labels) in splits.items():
    image_paths = []
    label_list = []

    split_image_dir = os.path.join(image_output_dir, split)
    os.makedirs(split_image_dir, exist_ok=True)

    for i, (image, label) in enumerate(zip(images, labels)):
        image_filename = f'{split}_image_{i}.png'
        image_filepath = os.path.join(split_image_dir, image_filename)

        save_image_array(image, image_filepath)

        image_paths.append(image_filepath)
        label_list.append(label)

    df = pd.DataFrame({'image_path': image_paths, 'label': label_list})
    csv_path = os.path.join(base_output_dir, f'{split}_dataset.csv')
    df.to_csv(csv_path, index=False)
    all_dataframes.append(df)

combined_df = pd.concat(all_dataframes, ignore_index=True)
combined_csv_path = os.path.join(base_output_dir, 'combined_dataset.csv')
combined_df.to_csv(combined_csv_path, index=False)

print(f"All saved")
