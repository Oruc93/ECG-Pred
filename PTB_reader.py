import numpy as np
import os
import scipy
import wfdb

def divide_chunks(l, n):
    """divides length of l into intervals of n until l is not long enough"""
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
#returns the ecg lead 1 in 2 second intervals
def load_data(path, sample_rate, T_or_T):
    """ Memorizes ecg data and preprocesses for injection into RNN preprocessing
    MIT-BIH time series are half hour long
    
    We read only MLII channels as ecg signal
    Additionally we construct a timeseries of R-peaks in samples with a samplerate of 256 Hz.
    For this we use the annotations time series
    
    ecgs have artifacts and are not suitable for training
    preprocessing needed in dataset construction
    
    Input:
        path: string. Path of ECG data
        sample_rate: int. samplerate for injection
        T_or_T: str. Training or Test set
        
    Output:
        data: np.array. ecg timeseries of 30min with 256Hz samplerate
        peaks: np.array. timeseries with r-peaks marked as int(3)
        train_BBI: list. BBI in ms
        """
    
    