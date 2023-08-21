from matplotlib import pyplot as plt
import pandas as pd
import math
import torch
import numpy as np
import h5py
import shutil
import os

from sequence import ECGSequence

def main():
    hdf5_data_path = r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part0.hdf5"
    csv_path = r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams.csv"

    # train_data, val_data = ECGSequence.get_train_and_val(
    #     r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part0.hdf5",
    #     "tracings",
    #     r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams.csv",
    #     batch_size=1)

    f = h5py.File(hdf5_data_path)
    train_data = np.array(f["tracings"])
    exam_id = np.array(f['exam_id'])
    
    dest_path = r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part0_imgs"
    shutil.rmtree(dest_path,   ignore_errors=True)
    os.makedirs(dest_path, exist_ok=True)

    for data, label in zip(train_data.transpose(0,2,1), exam_id):


        for i, signal in enumerate(data):

            plt.subplot(12,1,i+1)
            plt.plot(np.arange(0,2000), signal[648:2648])

            # Turn off spines and axis ticks
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().tick_params(axis='both', which='both', length=0)

            # Remove numbers along x and y-axes
            plt.gca().xaxis.set_major_formatter(plt.NullFormatter())
            plt.gca().yaxis.set_major_formatter(plt.NullFormatter())

        plt.savefig(os.path.join(dest_path, f'{label}.jpg'), bbox_inches="tight")
        plt.clf()
        print("done saving the id",label)
    

if __name__=="__main__":
    main()