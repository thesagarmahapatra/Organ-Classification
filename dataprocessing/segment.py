import os
import pandas as pd
import cv2

image_dir = "/content/drive/MyDrive/PneumoniaMNIST/images"
mask_dir = "/content/drive/MyDrive/PneumoniaMNIST/masks"
os.makedirs(mask_dir, exist_ok=True)

train_csv_path = "/content/drive/MyDrive/PneumoniaMNIST/train_dataset.csv"
val_csv_path = "/content/drive/MyDrive/PneumoniaMNIST/val_dataset.csv"
test_csv_path = "/content/drive/MyDrive/PneumoniaMNIST/test_dataset.csv"

train_updated_csv_path = "/content/drive/MyDrive/PneumoniaMNIST/train_dataset_with_masks.csv"
val_updated_csv_path = "/content/drive/MyDrive/PneumoniaMNIST/val_dataset_with_masks.csv"
test_updated_csv_path = "/content/drive/MyDrive/PneumoniaMNIST/test_dataset_with_masks.csv"

def generate_mask_with_opencv(image_path, threshold, mask_dir):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    mask_filename = os.path.basename(image_path).replace(".png", "_mask.png")
    mask_path = os.path.join(mask_dir, mask_filename)
    cv2.imwrite(mask_path, mask)
    return mask_path

def process_dataset_with_masks(csv_path, updated_csv_path, image_dir, mask_dir, threshold):
    df = pd.read_csv(csv_path)
    df['mask_path'] = df['image_path'].apply(
        lambda img_path: generate_mask_with_opencv(os.path.join(image_dir, os.path.basename(img_path)), threshold, mask_dir)
    )
    df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved to {updated_csv_path}")

threshold = 110

process_dataset_with_masks(train_csv_path, train_updated_csv_path, image_dir, mask_dir, threshold)
process_dataset_with_masks(val_csv_path, val_updated_csv_path, image_dir, mask_dir, threshold)
process_dataset_with_masks(test_csv_path, test_updated_csv_path, image_dir, mask_dir, threshold)

print(f"Masks generated and paths added to train, val, and test datasets.")
