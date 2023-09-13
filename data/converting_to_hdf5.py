import h5py
import wfdb
import numpy as np
import glob
import os
from argparse import ArgumentParser
import pandas as pd
import re
import csv
import sys
import logging

sys.path.append(r'C:\Users\ATI-G2\Documents\python\ECG')
from utils import wrappers

file_handler = logging.FileHandler(f"data_create_logs.txt",mode="w")
screen_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ", 
                    handlers=[file_handler, screen_handler] 
                    )


@wrappers.calculate_execution_time
def convert(limit: int):
    root_dir = args.ds
    glob_path = root_dir + r"\\*\\*\\RECORDS" 

    temp_arr = np.zeros((limit,3000,12),dtype=np.float16)
    k=0

    for i, pth in enumerate(glob.glob(glob_path)):

        with open(f"{pth}", "r") as g:
            records = g.readlines()
            for record in records:

                if k== limit:
                    with h5py.File(f'{args.dest_hdf5}',"w") as f:
                        f.create_dataset("tracings",data=temp_arr)

                    return

                try:
                    temp = pth
                    fil = record.replace("\n",'')
                    # print("hello",fil)
                    pth = pth.replace("RECORDS",str(fil))
                    signals, _ =  wfdb.rdsamp(pth, sampto=3000)
                    # print(signals.shape)
                    temp_arr[k,:,:] = signals
                    # print(temp_arr[0])

                except Exception as e:
                        logging.error(f'Exception cause in the record, {temp}, {record}, {k}, \n and the error is {e}')
                        # logging.error(e)
                        # sys.exit(0)


                print(k)
                k += 1
                pth = temp

@wrappers.calculate_execution_time
def convert_labels(limit):
    root_dir = args.ds
    glob_path = root_dir + r"\\*\\*\\RECORDS" 

    csv_pth = args.src_labels
    df = pd.read_csv(csv_pth)
    label_idx = {}

    all_labels = df["Snomed_CT"].values
    for i, label in enumerate(all_labels):
        label_idx[label] = i
        # idx_label[i] = label

    pattern = r"\d{3,}"

    # print("labelsl",len(all_labels))
    temp_arr = np.zeros((45000,len(all_labels)), dtype=np.int16)
    k = 0
    
    for i, pth in enumerate(glob.glob(glob_path)):
        with open(f"{pth}", "r") as g:
            
            records = g.readlines()
            for record in records:

                if k== limit:
                    with open(f"{args.dest_labels}","w",newline="\n") as f:
                        writer = csv.writer(f)
                        writer.writerows(temp_arr)

                    return

                fil = record.replace("\n",'')
                temp = pth
                pth = pth.replace("RECORDS",str(fil))
                try: 
                    _, fields =  wfdb.rdsamp(pth, sampto=3000)
                    string = fields["comments"][2]
                    # print("string--", string)
                    matches = re.findall(pattern, string)
                    # print("These are the matches \n",matches)
                    one_hot_vector = np.zeros((len(all_labels)),dtype=np.int16)
                    for j,match in enumerate(matches):
                        try:
                            one_hot_vector[label_idx[int(match.rstrip())]] = 1
                            # print(f"{match}-{i}")

                        except Exception as e:
                            print("exception",e)
                            pass
                    # print(i)
                    # print(one_hot_vector)
                    temp_arr[k,:]=one_hot_vector

                except Exception as e:
                    # print(f'Exception cause in the record, {temp}, {record}, {k}')
                    # print(e)
                    # sys.exit(0)
                    logging.error(f'Exception cause in the record, {temp}, {record}, {k}, \n and the error is {e}')


                pth = temp
                print(k)
                k +=1
                

    logging.info("temp_arr",temp_arr.shape)


    with open("{args.dest_labels}","w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(temp_arr)


    logging.info("done with the creating labels hdf5")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ds", default=r"E:\Chetan\ECG\data\12-lead\WFDBRecords")
    parser.add_argument("--src_labels", default=r"E:\Chetan\ECG\data\12-lead\ConditionNames_SNOMED-CT.csv")
    parser.add_argument("--dest_hdf5", default=r"12-lead.hdf5")
    parser.add_argument("--dest_labels", default=r"12-lead_labels.csv")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    limit = 45000
    # convert(limit)
    convert_labels(limit)