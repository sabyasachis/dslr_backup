import cv2
import numpy as np
import os, shutil
from tqdm import tqdm

sample_path = "./samples/"
merged_folder = "merged"

dynamic_folder = "dynamic"
reconstructed_folder = "reconstructed"
#static_folder = "static"

dynamic_path = os.path.join(sample_path, dynamic_folder)
reconstructed_path = os.path.join(sample_path, reconstructed_folder)
#static_path = os.path.join(sample_path, static_folder)
merged_path = os.path.join(sample_path, merged_folder)

def getint(name):
    return int(name.split('.')[0])

dynamic_files = sorted(os.listdir(dynamic_path), key=getint)
reconstructed_files = sorted(os.listdir(reconstructed_path), key=getint)
#static_files = sorted(os.listdir(static_path), key=getint)

if not os.path.exists(merged_path):
    os.makedirs(merged_path)
else:
    shutil.rmtree(merged_path)
    os.makedirs(merged_path)

for dynamic_file, reconstructed_file in tqdm(zip(dynamic_files, reconstructed_files), total=len(reconstructed_files)):
    dynamic_file_path = os.path.join(dynamic_path, dynamic_file)
    reconstructed_file_path = os.path.join(reconstructed_path, reconstructed_file)
    merged_file_path = os.path.join(merged_path, reconstructed_file)
    
    dynamic_img_arr = cv2.imread(dynamic_file_path)
    reconstructed_img_arr = cv2.imread(reconstructed_file_path)
    
    merged_img_arr = np.concatenate((dynamic_img_arr, reconstructed_img_arr), axis=1)
    cv2.imwrite(merged_file_path, merged_img_arr)
