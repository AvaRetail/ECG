'''
This file helps in converting the physioNet data format to hdf5 format;

Why am I doing this?
Ans) Because the model I want to run accepts the training data in hdf5 format.

parameters:
The data should be fo the shpae (n_samples, 4096, 12)
'''

import h5py
import wfdb
import numpy as np
import glob
from argparse import ArgumentParser
import pandas as pd
import re
import csv
import sys
import logging
from pathlib import Path

# importing functions from utils
sys.path.append(r'E:\Chetan\ECG')
from utils import wrappers
from utils.utils import increment_path

# creating folder for logs
save_dir = increment_path(Path("data\lovakant\exp") , mkdir=True)  # increment run


# logging
file_handler = logging.FileHandler(f"{save_dir}\data_create_logs.txt",mode="w")
screen_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ", 
                    handlers=[file_handler, screen_handler] 
                    )


'''
# @wrappers.calculate_execution_time
# def convert_to_npy(limit: int):
#     root_dir = args.ds
#     glob_path = root_dir + r"\\*\\*\\RECORD'
#     temp_arr = np.zeros((limit,4096,12),dtype=np.float16)
#     k=0

#     for i, pth in enumerate(glob.glob(glob_path)):

#         with open(f"{pth}", "r") as g:
#             records = g.readlines()
#             for record in records:

#                 if k== limit:
#                     # with h5py.File(f'{args.dest_hdf5}',"w") as f:
#                     #     f.create_dataset("tracings",data=temp_arr)
#                     np.save(f"{save_dir}/12-lead.npy", temp_arr)

#                     return

#                 try:
#                     temp = pth
#                     fil = record.replace("\n",'')
#                     # print("hello",fil)
#                     pth = pth.replace("RECORDS",str(fil))
#                     signals, _ =  wfdb.rdsamp(pth, sampto=4096)
#                     # print(signals.shape)
#                     temp_arr[k,:,:] = signals
#                     # print(temp_arr[0])

#                 except Exception as e:
#                         logging.error(f'Exception cause in the record, {temp}, {record}, {k}, \n and the error is {e}')
#                         # logging.error(e)
#                         # sys.exit(0)


#                 print(k)
#                 k += 1
#                 pth = temp
'''

@wrappers.calculate_execution_time
def convert_labels_to_npy():
    limit = args.limit
    root_dir = args.ds
    glob_path = root_dir + r"\\*\\*\\RECORDS" 
    csv_pth = args.src_labels

    df = pd.read_csv(csv_pth)
    dict_snomed_to_label = {}
    arr_snomed_label = df[["Snomed_CT", "HighLevelName", "Full Name"]].to_numpy()
    for i,val in enumerate(arr_snomed_label):
        dict_snomed_to_label[val[0]] = val[2]

    all_labels = args.filtered_labels

    # label_idx = {}
    # all_labels = df["HighLevelName"].unique()
    # for i, label in enumerate(all_labels):
    #     label_idx[label] = i

    labels_idx = {}
    for i, label in enumerate(all_labels):
        labels_idx[label] = i

    pattern = r"\d{3,}"

    # print("labelsl",len(all_labels))
    # temp_arr = np.zeros([1,4096,12], dtype=np.float16)
    # temp_arr_labels = np.zeros([1, len(all_labels)], dtype=np.int16)
    
    temp_arr = np.zeros((24000,4096,12),dtype=np.float16)
    temp_arr_labels = np.zeros((24000,len(all_labels)), dtype=np.int16)
    k = 0
    m = 0

    for i, pth in enumerate(glob.glob(glob_path)):
        with open(f"{pth}", "r") as g:
            
            records = g.readlines()
            for record in records:

                if k== limit:

                    # with open(f"{args.dest_labels}","w",newline="\n") as f:
                    #     writer = csv.writer(f)
                    #     writer.writerows(temp_arr)

                    np.save(f"{args.dest_labels_pth}", temp_arr_labels)
                    np.save(f"{args.dest_data_pth}", temp_arr)
                    print("===Done saving both the labels and data in the npy format===")
                    return

                fil = record.replace("\n",'')
                temp = pth
                pth = pth.replace("RECORDS",str(fil))
                try: 
                    signals, fields =  wfdb.rdsamp(pth, sampfrom=500,sampto=4596)
                    string = fields["comments"][2]
                    matches = re.findall(pattern, string)
                    for j,match in enumerate(matches):
                        try:
                            label = dict_snomed_to_label[int(match.rstrip())]
                            if label in args.filtered_labels:

                                one_hot_vector = np.zeros((len(all_labels)), dtype=np.int16)
                                one_hot_vector[labels_idx[label]] = 1
                                # temp_arr_labels.append(one_hot_vector)
                                # temp_arr.append(signals)
                                temp_arr[m,:,:] = signals
                                temp_arr_labels[m, :] = one_hot_vector
                                print(f"{label} is there")
                                label = None
                                m +=1 #
                                print(m)


                        except Exception as e:
                            print("exception",e)
                            label = None
                            pass


                except Exception as e:
                    logging.error(f'Exception cause in the record, {temp}, {record}, {k}, \n and the error is {e}')


                pth = temp
                k = k+1
                
                

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ds", default=r"E:\Chetan\ECG\data\lovakant\WFDBRecords")
    parser.add_argument("--src_labels", default=r"E:\Chetan\ECG\data\lovakant\ConditionNames_SNOMED-CT.csv")
    parser.add_argument("--filtered_labels", default=["Sinus Rhythm", "Atrial Flutter", "Sinus Tachycardia"], type=str, nargs="+")
    parser.add_argument("--limit", default=45131, type=int)
    parser.add_argument("--dest_data_pth", default=f"{save_dir}/12-lead.npy")
    parser.add_argument("--dest_labels_pth", default=f"{save_dir}/12-lead-labels.npy")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    convert_labels_to_npy()