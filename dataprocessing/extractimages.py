import numpy as np
import os
from PIL import Image
import pandas as pd

data = np.load('breastmnist_224.npz')
train_images = data['train_images']
train_labels = data['train_labels']

output_dir = 'BreastMNIST_Folder'
images_dir = os.path.join(output_dir, 'images')

os.makedirs(images_dir, exist_ok=True)

data_list = []

for i in range(train_images.shape[0]):
    image = train_images[i]
    label = train_labels[i]
    
    image_pil = Image.fromarray(image)
    image_filename = f'image_{i}.png'
    image_path = os.path.join(images_dir, image_filename)
    image_pil.save(image_path)

    label_list = [image_filename] + label.tolist() 
    data_list.append(label_list)

column_names = ['filename'] + [f'label_{i}' for i in range(train_labels.shape[1])]
df = pd.DataFrame(data_list, columns=column_names)

csv_path = os.path.join(output_dir, 'labels.csv')
df.to_csv(csv_path, index=False)

print('Images have been saved to the Images directory and labels have been saved to labels.csv')
