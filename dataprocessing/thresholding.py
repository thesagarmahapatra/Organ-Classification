import cv2 as cv
import os

def process_and_save_thresholded_images(input_dir, output_dir):
    thresOpt = [cv.THRESH_TOZERO]
    thresNames = ['toZero']
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(input_path):
            continue
        img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping file {file_name}: not a valid image")
            continue

        for i in range(len(thresOpt)):
            _, imgThres = cv.threshold(img, 125, 255, thresOpt[i])
            
            base_name, ext = os.path.splitext(file_name)
            output_file_name = f"{base_name}_{thresNames[i]}{ext}"
            output_path = os.path.join(output_dir, output_file_name)
            
            cv.imwrite(output_path, imgThres)
            print(f"Saved: {output_path}")

if __name__ == '__main__':
    input_directory = '/Users/ssm/Code/FML Project/images'
    output_directory = '/Users/ssm/Code/FML Project/thresholded_images'
    process_and_save_thresholded_images(input_directory, output_directory)
