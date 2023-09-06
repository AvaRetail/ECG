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

sys.path.append(r'C:\Users\ATI-G2\Documents\python\ECG')
from utils import wrappers

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
                    with h5py.File(f'12-lead.hdf5',"w") as f:
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
                        print(f'Exception cause in the record, {temp}, {record}, {k}')
                        print(e)

                print(k)
                k += 1
                pth = temp

            
def convert_labels():
    root_dir = args.ds
    glob_path = root_dir + r"\\*\\*\\RECORDS" 

    csv_pth = r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\12-lead\ConditionNames_SNOMED-CT.csv"
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
                    print(f'Exception cause in the record, {temp}, {record}, {k}')

                pth = temp
                print(k)
                k +=1
                

    print("temp_arr",temp_arr.shape)

    # with h5py.File("12-lead_labels.hdf5", "w") as f:
    #     f.create_dataset("labels",data=temp_arr)

    with open("12-lead_labels.csv","w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(temp_arr)


    print("done with the creating labels hdf5")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ds", default=r"C:\Users\ATI-G2\Documents\python\ECG\data\12-lead\WFDBRecords")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    convert(45000)
    # convert_labels()