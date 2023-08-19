from matplotlib import pyplot as plt
import pandas as pd
import math
import torch
import numpy as np
import h5py

from sequence import ECGSequence

def main():
    train_data, val_data = ECGSequence.get_train_and_val(r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part0.hdf5",
"tracings",r"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams.csv",batch_size=1)
    
    for (data,_), label in zip(train_data, train_data.exam_id):


        for i, signal in enumerate(data[0]):

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
        
    # plt.show()
        plt.savefig(fr"C:\Users\ATI-G2\Documents\python\ECG\data\code-15\exams_part0_imgs\{label}.jpg", bbox_inches="tight")
        plt.clf()
        print("done saving the id",label)
    

if __name__=="__main__":
    main()