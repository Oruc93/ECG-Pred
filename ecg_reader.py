import numpy as np
import os
import scipy
import wfdb

def divide_chunks(l, n):
    """divides length of l into intervals of n until l is not long enough"""
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
#returns the ecg lead 1 in 2 second intervals
def load_data(path, sample_rate, T_or_T, duration=2**17/1024):
    """ Memorizes ecg data and preprocesses for injection into RNN preprocessing
    MIT-BIH time series are half hour long
    
    We read only MLII channels as ecg signal
    Additionally we construct a timeseries of R-peaks in samples with a samplerate of 256 Hz.
    For this we use the annotations time series
    
    
    Input:
        path: string. Path of ECG data
        sample_rate: int. samplerate for injection
        duration: int. duration of time series in s
        number_of_files: int. number of ecg time series extracted from data source"""
    # if os.path.exists('temp/train_data.npy'):
    #     print("Found cached training data")
    #     return np.load('temp/train_data.npy', allow_pickle = True)
    train_data = []
    train_beats = []
    train_BBI = []
    # check if ecg is clean!!!!
    example = 0 # count number of examples
    for i in range(134): # loop over all time series files 134
        if T_or_T == 'training' and example >= 0.8*45: # check if enough examples for training
            continue
        
        hea_path = "/{}".format(str(i+100).zfill(2))
        dat_path = "/{}.dat".format(str(i+100).zfill(2))
        try: # try, if file is existing and if it has MLII lead, continue, if not
            # record = ECGRecord.from_wfdb(path + hea_path)
            signals, fields = wfdb.rdsamp(path + hea_path) # read physical signals
            lead_1 = signals[:,fields["sig_name"].index('MLII')] # extract MLII lead
            ann = wfdb.rdann(path + hea_path, 'atr', return_label_elements='label_store') # read annotations
            beats = np.zeros(np.shape(lead_1)) # construct baseline timeseries
            # ann.label_store enthält die Annotationen des EKG. 
            # wfdb.show_ann_labels() # Liste der Labels und deren Bedeutung
            # ann.sample enthält die Positionen der Annotationen in sample
            # beat_labels = ann.label_store>=1 & ann.label_store<=13 & ann.label_store!=5 & ann.label_store!=8
            beat_labels = np.all(np.stack([ann.label_store>=1, ann.label_store<=13, ann.label_store!=5, ann.label_store!=8]), axis=0)
            # print(np.shape(beat_labels))
            beat_sample = ann.sample[beat_labels] # positions of beats in samples
            BBI = (beat_sample[1:] - beat_sample[:-1])/360*1000 # construct BBI in ms
            beats[beat_sample] = 3 # timeseries of beats marked as 3 in samples with 256 Hz. Constr_feat 'Tacho' case looks for 3
            # print(hea_path)
            # print(fields["sig_name"])
            # print(fields["sig_name"].index('MLII'))
            
            # print(ann.sample)
            # print(np.unique(ann.label_store))
            # print(np.shape(ann.sample))
            # print(np.shape(ann.label_store))
            # wfdb.show_ann_classes()
            # wfdb.show_ann_labels()
            # signal in signals
            # rebuild preparation with wfdb
            
            
            if T_or_T == 'test' and example < 0.8*45:
                example += 1
                continue
            # annotation = record.annotations()
            example += 1
        except:
            # print(hea_path, " not found")
            continue

        # chunked = list(divide_chunks(lead_1, int(duration*360)))
        # chunked_b = list(divide_chunks(beats, int(duration*360)))
        # chunk_data = []
        # chunk_data_b = []
        # n = 0
        # for chunk in chunked:
        #     # append interval if chunk is fully formed
        #     if len(chunk) == duration*360:
        #         chunk_data.append( (chunk - np.mean(chunk) / max(chunk - np.mean(chunk))) )
        #         chunk_data_b.append(chunked_b[n])
        #     n += 1
        # # choose one of the fully formed chunks randomly
        # # print(len(chunk_data[0]))
        # example = np.random.randint(len(chunk_data))
        # train_data.append(chunk_data[example])
        # train_beats.append(chunk_data_b[example])
        train_data.append(lead_1)
        train_beats.append(beats)
        train_BBI.append(BBI)
        
    #train_data = [(j - np.mean(j)) / max(j - np.mean(j)) for i in patient_chunks for j in i]
    
    # train_data = [x for x in train_data if (np.std(x) < 0.3)]
    # train_data = np.array(chunk_data) # transform to numpy array
    
    # downsample data from 360Hz to 256Hz
    # by resampling time series with number of point you get with duration multiplied with desired samplerate
    # tested, it is good, if last value is similar to first (periodically), because we use FFT
    # data = np.array([scipy.signal.resample(d, int(duration*sample_rate)) for d in train_data])
    data = np.array([scipy.signal.resample(d, 1+int(len(train_data[0])/360*sample_rate)) for d in train_data])
    peaks = []
    step = 360/sample_rate # window size for downsampling
    for d in train_beats: # loop over examples
        index = 0 # index of original beats timeseries
        n = 0 # index of downsampled beats timeseries
        d_down = np.zeros(np.shape(data[0,:]))
        # print(np.shape(data[0,:]))
        # print(len(d))
        # print(np.unique(d))
        while index<len(d):
            d_down[n] = max(d[int(index):int(index + step)])
            # print(d[int(index):int(index + step)])
            index += step
            n += 1
            # print(index)
            
        # print(np.unique(d_down))
        peaks.append(d_down)
    peaks = np.array(peaks)
    # print(np.shape(peaks))
    #save processed data to minimize load time
    # if not os.path.exists('temp/'):
    #     os.makedirs('temp')
    # print("saving training data")
    # np.save('temp/train_data.npy', train_data, allow_pickle = True)
    
    #preprocess data
    # train_data = []
    # for lead in raw_data:
        # lead = lead - np.mean(lead)
        # lead = lead / max(lead)
        # data.append(lead)
        
    return data, peaks, train_BBI