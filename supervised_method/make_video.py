import cv2
import numpy as np
import os, shutil
from tqdm import tqdm


sample_path = "./samples/"
merged_folder = "merged"
output_path = os.path.join(sample_path, merged_folder+".avi")

merged_path = os.path.join(sample_path, merged_folder)
def getint(name):
    return int(name.split('.')[0])

merged_files = sorted(os.listdir(merged_path), key=getint)

some_file_path = os.path.join(merged_path, merged_files[0])
some_img = cv2.imread(some_file_path)
height, width, layers = some_img.shape

video_size = (width,height)
video_fps  = 10

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), video_fps, video_size)

for merged_file in tqdm(merged_files):
    merged_file_path = os.path.join(merged_path, merged_file)
    merged_img_arr = cv2.imread(merged_file_path)
    out.write(merged_img_arr)
    
out.release()