import numpy as np
import os
import scipy
import wfdb

# wfdb docs
# https://wfdb.readthedocs.io/en/latest/io.html#module-1

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
    train_data = []
    train_beats = []
    train_BBI = []
    
    example = 0 # count number of succesful examples
    for i in range(134): # loop over all time series files 134
        if T_or_T == 'training' and example >= 0.8*45: # check if enough examples for training
            continue
        
        hea_path = "/{}".format(str(i+100).zfill(2))
        try: # try, if file is existing and if it has MLII lead, continue, if not
            
            signals, fields = wfdb.rdsamp(path + hea_path) # read physical signals
            lead_1 = signals[:,fields["sig_name"].index('MLII')] # extract MLII lead
            
            ann = wfdb.rdann(path + hea_path, 'atr', return_label_elements='label_store') # read annotations
            
            # array with labels of interest
            # wfdb.show_ann_labels() # list of labels and their meaning
            beat_labels = np.all(np.stack([ann.label_store>=1, ann.label_store<=13, ann.label_store!=5, ann.label_store!=8]), axis=0)
            
            beat_sample = ann.sample[beat_labels] # positions of beats in samples
            BBI = (beat_sample[1:] - beat_sample[:-1])/360*1000 # construct BBI in ms
            
            beats = np.zeros(np.shape(lead_1)) # construct baseline timeseries
            beats[beat_sample] = 3 # timeseries of beats marked as 3 in samples with 256 Hz. Constr_feat 'Tacho' case looks for 3
            
            if T_or_T == 'test' and example < 0.8*45: # check if samples for tests is reached
                example += 1
                continue
            
            example += 1
        except:
            continue

        train_data.append(lead_1)
        train_beats.append(beats)
        train_BBI.append(BBI)
    
    # downsample data from 360Hz to 256Hz
    # by resampling time series with number of point you get with duration multiplied with desired samplerate
    # tested, it is good, if last value is similar to first (periodically), because we use FFT
    data = np.array([scipy.signal.resample(d, 1+int(len(train_data[0])/360*sample_rate)) for d in train_data])
    peaks = []
    step = 360/sample_rate # window size for downsampling
    for d in train_beats: # loop over examples
        index = 0 # index of original beats timeseries
        n = 0 # index of downsampled beats timeseries
        d_down = np.zeros(np.shape(data[0,:])) # array for downsampled timeseries
        while index<len(d):
            d_down[n] = max(d[int(index):int(index + step)])
            index += step
            n += 1
            
        peaks.append(d_down)
    peaks = np.array(peaks)
        
    return data, peaks, train_BBI