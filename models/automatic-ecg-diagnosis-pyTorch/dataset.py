import h5py
import math
import pandas as pd
# from tensorflow.keras.utils import Sequence
import numpy as np
import torch


class ECGSequence():
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, split=0.1):
        
        n_samples = len(pd.read_csv(path_to_csv).to_numpy())

        val_split = split
        
        train_split = 1- 2*val_split
        n_train = int(n_samples*train_split)
        n_val = int(n_samples*val_split)

        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx = 0, end_idx = n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx = n_train, end_idx = n_train+n_val)
        test_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx = n_train+n_val, end_idx = n_samples)
        return train_seq, valid_seq, test_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.end_idx = end_idx
        self.start_idx = start_idx
        self.n_batches = math.floor((self.end_idx-self.start_idx)/self.batch_size)
        self.idx = 0
        # self.f.close()

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __next__(self):
        if self.idx == self.n_batches:
            raise StopIteration
        else: 
            start = self.start_idx + self.idx * self.batch_size
            end = min(start + self.batch_size, self.end_idx)
            self.idx += 1
            return torch.Tensor(self.x[start:end, :, :]).permute(0,2,1), torch.Tensor(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()

    def __iter__(self):
        self.idx = 0
        return self

