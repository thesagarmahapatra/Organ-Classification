import numpy as np
import pandas as pd
import os
from PIL import Image

dataset_path = 'octmnist_224.npz' 
data = np.load(dataset_path)

train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']

def save_image_array(image_array, output_path):
    image = Image.fromarray(image_array.astype('uint8'))  
    image.save(output_path)


output_dir = '/Users/ssm/Code/OCTMNIST/images'  
os.makedirs(output_dir, exist_ok=True)

def sample_indices(array_length, sample_fraction):
    return np.random.choice(array_length, size=int(array_length * sample_fraction), replace=False)

def sample_and_process_images(images, labels, dataset_type, sample_fraction=0.1):
    sampled_indices = sample_indices(len(images), sample_fraction)
    sampled_images = images[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    image_paths = []
    label_list = []

    for i, (image, label) in enumerate(zip(sampled_images, sampled_labels)):
        image_filename = f'{dataset_type}_image_{i}.png'  # Create a unique filename
        image_filepath = os.path.join(output_dir, image_filename)

        # Save image to file
        save_image_array(image, image_filepath)

        image_paths.append(image_filepath)
        label_list.append(label)

    return image_paths, label_list

train_image_paths, train_labels_list = sample_and_process_images(train_images, train_labels, 'train')
val_image_paths, val_labels_list = sample_and_process_images(val_images, val_labels, 'val')
test_image_paths, test_labels_list = sample_and_process_images(test_images, test_labels, 'test')

train_df = pd.DataFrame({'image_path': train_image_paths, 'label': train_labels_list})
val_df = pd.DataFrame({'image_path': val_image_paths, 'label': val_labels_list})
test_df = pd.DataFrame({'image_path': test_image_paths, 'label': test_labels_list})

train_csv_path = 'train_dataset_sampled.csv'
val_csv_path = 'val_dataset_sampled.csv'
test_csv_path = 'test_dataset_sampled.csv'

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

combined_csv_path = 'combined_dataset_sampled.csv'
combined_df.to_csv(combined_csv_path, index=False)

print(f"Sampled images saved and CSV files created for train, validation, and test datasets.")
print(f"Sampled train dataset saved to: {train_csv_path}")
print(f"Sampled validation dataset saved to: {val_csv_path}")
print(f"Sampled test dataset saved to: {test_csv_path}")
print(f"Combined dataset saved to: {combined_csv_path}")
