'''
Script-1 in the pre-processing pipeline

This script converts the signal from hdf5 to the images and stores them in the folder named same as the hdf5 file
'''

import numpy as np
import h5py
import shutil
import os
from matplotlib import pyplot as plt
from sequence import loadHdf5
from argparse import ArgumentParser


def generate_image(data, label):
    plt.figure(figsize=(20, 15))  # Adjust the width and height as needed

    for i, signal in enumerate(data.T):

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

    # plt.savefig(os.path.join(dest_path, f'{label}.jpg'), bbox_inches="tight")
    plt.savefig(f'{label}.jpg', bbox_inches="tight")
    plt.show()
    plt.clf()
    print("done saving the id",label)


def generate_folders(dest):
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)

def main():

    generate_folders(dest_path)
    data_model = loadHdf5(hdf5_data_path)
    size = np.array(data_model.ids).shape[0]

    for i in range(size):
        data, label = data_model[i]
        generate_image(data, label)
        break

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dest_path", default=r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part5_imgs")
    parser.add_argument("--hdf5_data_path", default=r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part5.hdf5") 

    return parser.parse_args()

if __name__=="__main__":

    args = get_args()
    dest_path = args.dest_path
    hdf5_data_path = args.hdf5_data_path
    main()