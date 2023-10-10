import numpy as np
import h5py
import csv
import glob
import os

src_dir = r"E:\Chetan\ECG\data\lovakant\exp4"
np.random.seed(0)

labels = np.load(os.path.join(src_dir, "12-lead-labels.npy"))[:23300]
data = np.load(os.path.join(src_dir, "12-lead.npy"))[:23300]
n_samples =labels.shape[0]

samples_order =  np.arange(n_samples)

np.random.shuffle(samples_order)

with h5py.File(os.path.join(src_dir, "12-lead.hdf5"),"w") as f:
    f.create_dataset("tracings", data=data[samples_order])

print("---Done with creating the hdf5 file--")

with open(os.path.join(src_dir, "12-lead-labels.csv"), "w",newline="\n") as f:
    writer = csv.writer(f)
    writer.writerows(labels[samples_order])

print("---Created the csv file---")


    
