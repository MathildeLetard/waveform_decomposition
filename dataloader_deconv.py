import scipy
import numpy as np
import os
from default_deconv import *
import h5py

class DataLoader():
    def __init__(self, ndataset):
        self.ndataset = ndataset
        self.dataset=h5py.File(ndataset,'r')
        self.nb_ech =self.dataset['waveforms'].shape[0]

    def load_batch(self,batch_size):
        ibatch = np.sort(np.random.choice(self.nb_ech, size=batch_size, replace=False))
        X = self.dataset['waveforms'][ibatch]
        X = X.reshape((X.shape[0], DEF_LEN, 1))
        Y = self.dataset['decomp'][ibatch]
        Y = [Y[:,:,0].reshape((X.shape[0],DEF_LEN,1)), Y[:,:,1].reshape((X.shape[0],DEF_LEN,1)), Y[:,:,2].reshape((X.shape[0],DEF_LEN,1))]
        return X,Y

    def load_data(self):
        X = self.dataset['waveforms'][:]
        X = X.reshape((X.shape[0], DEF_LEN, 1))
        Y = self.dataset['decomp'][:]
        Y = [Y[:,:,0].reshape((X.shape[0],DEF_LEN,1)), Y[:,:,1].reshape((X.shape[0],DEF_LEN,1)), Y[:,:,2].reshape((X.shape[0],DEF_LEN,1))]
        return X,Y

    def load_val_batch(self,batch_size,niter):
        X = self.dataset['waveforms'][niter*batch_size:niter*batch_size + batch_size]
        X = X.reshape((X.shape[0],DEF_LEN,1))
        Y = self.dataset['decomp'][niter*batch_size:niter*batch_size + batch_size]
        Y = [Y[:,:,0].reshape((X.shape[0],DEF_LEN,1)), Y[:,:,1].reshape((X.shape[0],DEF_LEN,1)), Y[:,:,2].reshape((X.shape[0],DEF_LEN,1))]
        return X,Y

    def load_test_data(self):
        X = self.dataset['waveforms'][:]
        X = X.reshape((X.shape[0], DEF_LEN, 1))
        return X

    def load_test_labels(self):
        Y = self.dataset['decomp'][:]
        Y = [Y[:,:,0], Y[:,:,1], Y[:,:,2]]
        return Y

    def load_testbatch(self,batch_size,niter):
        X = self.dataset['waveforms'][niter*batch_size:niter*batch_size + batch_size]
        X = X.reshape((X.shape[0], DEF_LEN, 1))
        return X

    def close(self):
        self.dataset.close()
