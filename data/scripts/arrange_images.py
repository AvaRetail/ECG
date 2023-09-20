'''
Script-2 in the data pre-processing pipeline

This script helps in moving the images that were generted to the vit-data.

After running this script use the train_test_val.py script.
'''

import numpy as np
import os
import shutil
import re
import glob
import pandas as pd
import h5py

hdf5_path =  r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part5.hdf5" #-- change this line
exams_csv_path = r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams.csv"

hdf5_path =  r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part5.hdf5" #-- change this line

# with h5py.File(hdf5_path,"r") as f:

pd_results = pd.read_csv(exams_csv_path,index_col="exam_id")

file_name = re.split(r"\\", hdf5_path)[-1]

labels_list = ["1dAVb","RBBB","LBBB","SB","ST","AF","normal_ecg", "not_normal_ecg"]

pd_req = pd_results[pd_results["trace_file"]==file_name][labels_list[:-1]]
labels_arr = pd_req.to_numpy()
exam_id = pd_req.index.to_list()

dest_root_fldr = r"C:\Users\ATI-G2\Documents\python\ECG\data\rough"
src_path = r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part5_imgs" #--Change this line
normal_count = 0
not_normal_count = 0
disease_count = 0

shutil.rmtree(dest_root_fldr, ignore_errors = True)
os.makedirs(dest_root_fldr, exist_ok = True)

for flr in labels_list:
    os.makedirs(os.path.join(dest_root_fldr, f'{flr}'))


for i,idx in enumerate(exam_id):
    try:
        if labels_arr[i,6]==1:
            shutil.copy(os.path.join(src_path,f"{idx}.jpg"),os.path.join(dest_root_fldr, "normal_ecg"))
            normal_count += 1

        else:
            if np.count_nonzero(labels_arr[i])==0:
                shutil.copy(os.path.join(src_path,f"{idx}.jpg"), os.path.join(dest_root_fldr, "not_normal_ecg"))
                not_normal_count +=1

            else:
                clas = labels_list[np.where(labels_arr[i]==1)[0][0]]
                shutil.copy(os.path.join(src_path,f'{idx}.jpg'), os.path.join(dest_root_fldr, f"{clas}"))
                disease_count += 1

    except Exception as e:
        print(e)

print("normal count", normal_count)
print("not normal count", not_normal_count)
print("disease count", disease_count)


src = r"C:\Users\ATI-G2\Documents\python\ECG\data\rough"
dest = r"C:\Users\ATI-G2\Documents\python\ECG\data\vit-data"

for path in glob.glob(src+r"\\*\\*"):
    sub_dir = re.split(r"\\", path)[-2]
    if not "ecg" in sub_dir:
        shutil.copy(path, os.path.join(dest, sub_dir))
