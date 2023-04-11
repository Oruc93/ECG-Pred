import wfdb
import sys, os
import numpy as np
import time
import random
import multiprocessing as mp

# wfdb docs
# https://wfdb.readthedocs.io/en/latest/io.html#module-1

#example from databse page
# patient_id=9000
# segment_id=0
# start=2000
# length=1024
# filename = f'{data_path}/p0{str(patient_id)[:1]}/p{patient_id:05d}/p{patient_id:05d}_s{segment_id:02d}'
# rec = wfdb.rdrecord(filename, sampfrom=start, sampto=start+length)
# ann = wfdb.rdann(filename, "atr", sampfrom=start, sampto=start+length, shift_samps=True)
# wfdb.plot_wfdb(rec, ann, plot_sym=True, figsize=(15,4));

"""
This library is for downloading ecgs from Icentia11k database
https://physionet.org/content/icentia11k-continuous-ecg/1.0/

selecting ecgs for training and testing
downloading ecgs and annotations singularly, because whole dataset is huge
formatting for own usage
feeding into neural network preprocessing"""

# changing current working folder to directory of script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
local_path = os.getcwd() + '/data/Icentia11k/'

def download(amount):
    """ List all files of interest seperated in training and test
    Downloads ecgs and annotations of interest
    
    Input:
        amount: int. number of files for training"""
    
    print("Preparing list of segments ...")
    list_segments = set() # list of all segments
    list_patients = set() # set of all patients
    # Loading ever expanding set of existing segments
    if os.path.exists(local_path + "set_records.txt"):
        print("Load set records ...")
        with open(local_path + "set_records.txt", 'r') as f:
            lines = f.readlines()
            for r in range(len(lines)):
                list_segments.add(lines[r].rstrip('\n'))
                list_patients.add(lines[r][:-5].rstrip('\n'))
    print("Downloading training files ...")
    print("Listing available segments of interest ...")
    check_download = True
    while check_download: # loop for failed download
        file_list_training = []
        for n in range(amount): # loop over amount of needed files for training
            check = True
            while check: # loop until unused path found
                patient_id = np.random.randint(11000) # pull random patient from 11.000
                ecg_id = np.random.randint(50) # pull random ecg from 50
                file_path = 'p' + f"{patient_id:05d}"[:2] + '/p' + f"{patient_id:05d}" + '/p' + f"{patient_id:05d}" + '_s' + f"{ecg_id:02d}"
                # print(file_path)
                if not(file_path in file_list_training): # check if file is already selected
                    # Download list of available segments of patient
                    # Check if table of segments for patient is already downloaded. Saving time in the long run
                    # if any([True for s in list_segments if '/p' + f"{patient_id:05d}" in s]):
                    if 'p' + f"{patient_id:05d}" in list_patients:
                        print("Already downloaded")
                    else:
                        # print("Fresh download")
                        sys.stdout = open(os.devnull, 'w') # blocks print
                        wfdb.dl_files('icentia11k-continuous-ecg/1.0/',
                                        local_path,
                                        ['p' + f"{patient_id:05d}"[:2] + '/p' + f"{patient_id:05d}" + '/RECORDS'],
                                        overwrite=True, # skips already downloaded files
                                        keep_subdirs=False
                                        )
                        sys.stdout = sys.__stdout__
                        with open(local_path + 'RECORDS') as f: # load table of segments as set
                            records = f.readlines()
                            for r in range(len(records)):
                                list_segments.add(records[r].rstrip('\n'))
                                list_patients.add(lines[r][:-5].rstrip('\n'))
                    if 'p' + f"{patient_id:05d}" + '_s' + f"{ecg_id:02d}" in list_segments: # check if selected segment exists in records
                        file_list_training.append(file_path)
                        # print("Segment found")
                        # print('p' + f"{patient_id:05d}" + '_s' + f"{ecg_id:02d}" + " segment exists")
                        check = False
                
        # download files of interest
        try:
            dl_list = []
            for file in file_list_training:
                dl_list.append(file + '.atr')
                dl_list.append(file + '.hea')
                dl_list.append(file + '.dat')
            wfdb.dl_files('icentia11k-continuous-ecg/1.0/',
                            local_path,
                            dl_list,
                            # [file + '.atr' for file in file_list_training]+[file + '.hea' for file in file_list_training]+[file + '.dat' for file in file_list_training],
                            overwrite=False, # skips already downloaded files
                            keep_subdirs=False
                            )
            check_download = False
        except:
            print("A selected segment for training did not exist in database.")
            print(sorted(file_list_training))
            print(file_list_training[0]+'.atr')
            print('Retrying')

    print("Downloading test files ...")
    print("Listing available segments of interest ...")
    check_download = True
    while check_download: # loop for failed download
        file_list_test = []
        for n in range(int(max([1,amount*0.2]))):
            check = True
            while check: # loop until unused path found
                patient_id = np.random.randint(11000) # pull random patient from 11.000
                ecg_id = np.random.randint(50) # pull random ecg from 50
                file_path = 'p' + f"{patient_id:05d}"[:2] + '/p' + f"{patient_id:05d}" + '/p' + f"{patient_id:05d}" + '_s' + f"{ecg_id:02d}"
                # print(file_path)
                if not(file_path in file_list_test): # check if file is already selected
                    if not(any([f"{patient_id:05d}" in path for path in file_list_training])): # check if patient is already in training set
                        # Download list of available segments of patient
                        # Check if table of segments for patient is already downloaded. Saving time in the long run
                        if any([True for s in list_segments if '/p' + f"{patient_id:05d}" in s]):
                            print("Already downloaded")
                        else:
                            # print("Fresh download")
                            sys.stdout = open(os.devnull, 'w') # blocks print
                            wfdb.dl_files('icentia11k-continuous-ecg/1.0/',
                                            local_path,
                                            ['p' + f"{patient_id:05d}"[:2] + '/p' + f"{patient_id:05d}" + '/RECORDS'],
                                            overwrite=True, # skips already downloaded files
                                            keep_subdirs=False
                                            )
                            sys.stdout = sys.__stdout__
                            with open(local_path + 'RECORDS') as f: # load table of segments as set
                                records = f.readlines()
                                for r in range(len(records)):
                                    list_segments.add(records[r].rstrip('\n'))
                        if 'p' + f"{patient_id:05d}" + '_s' + f"{ecg_id:02d}" in list_segments: # check if selected segment exists in records
                            file_list_test.append(file_path)
                            # print("Segment found")
                            # print('p' + f"{patient_id:05d}" + '_s' + f"{ecg_id:02d}" + " segment exists")
                            check = False
        # download files of interest
        try:
            dl_list = []
            for file in file_list_test:
                dl_list.append(file + '.atr')
                dl_list.append(file + '.hea')
                dl_list.append(file + '.dat')
            wfdb.dl_files('icentia11k-continuous-ecg/1.0/',
                            local_path,
                            dl_list,
                            # [file + '.atr' for file in file_list_test]+[file + '.hea' for file in file_list_test]+[file + '.dat' for file in file_list_test],
                            overwrite=False, # skips already downloaded files
                            keep_subdirs=False
                            )
            check_download = False
        except:
            print("A selected segment for test did not exist in database")
            print(sorted(file_list_training))
            print(file_list_training[0]+'.atr')
            print('Retrying')
            
    # print(file_list_training)
    # print(file_list_test)
    
    # save set of available segments in txt-file
    with open(local_path + "set_records.txt", 'w') as f:
            f.write('\n'.join(list_segments))
    
    return file_list_training, file_list_test

def load_local(amount):
    """ List all files of interest seperated in training and test
    Only locally saved segments
    
    Input:
        amount: int. number of files for training"""
    
    print("Preparing list of segments ...")
    # Load locally available segments and patients
    list_segments = [] # list of all segments
    list_patients = set()
    file_dir = os.listdir(local_path)
    for file in file_dir:
        if ".hea" in file: # check if file is 
            if file[:-4] + ".atr" in file_dir and file[:-4] + ".dat" in file_dir:
                list_segments.append(file[:-4])
                list_patients.add(file[:6])
    print("Number of patients: ", len(list_patients))
    print("Number of segments: ", len(list_segments))
    # exit()
    print("Listing available segments of interest ...")
    file_list_training = []
    for n in range(amount): # loop over amount of needed files for training
        check = True
        while check: # loop until unused path found
            file_path = np.random.choice(list_segments) # pull file path of random segment
            list_segments.remove(file_path)
            if not(file_path in file_list_training): # check if file is already selected
                file_list_training.append(file_path)
                check = False

    print("Listing available segments of interest ...")
    file_list_test = []
    counter = 0
    missing_test = 0 # counter for how many test patients couldnt be found
    for n in range(int(max([1,amount*0.2]))):
        check = True
        while check: # loop until unused path found
            file_path = np.random.choice(list_segments) # pull file path of random segment
            list_segments.remove(file_path)
            # print(file_path)
            if not(file_path in file_list_test): # check if file is already selected
                if not(any([file_path[:6] in path for path in file_list_training])): # check if patient is already in training set
                    file_list_test.append(file_path)
                    check = False
                else:
                    counter += 1
                if counter > 999: # check if list of non-training patients is exhausted
                    missing_test += 1
                    print("No patients left for test. Need ", str(int(amount*0.2-missing_test)), " more test patients")
                    break
    
    return file_list_training, file_list_test

def load_data(list_training, length_item):
    """ This function outdated and not used anymore
    
    Loads ecgs and annotation of selected segments
    Segments are downloaded beforehand
    Segments are cut to shorter length for processing efficiency
    
    Input:
        list of strings. list of selected segments
        
    Return:
        data. numpy.array of floats. ecgs
        peaks. numpy.array of int. positions of r-peaks marked by int(3)
        BBI. list of arrays. int. beat-to-beat intervalls"""
    train_data = []
    train_beats = []
    train_BBI = []
    
    for file in list_training:
        signals, fields = wfdb.rdsamp(local_path + file[-11:]) # read physical signals
        # cut off first two seconds and extract ecg of desired length + 2 seconds added to end
        # signals is in 250Hz
        # length_item is in 250Hz. calculate accordingly
        ecg = signals[1000:1000+500+int(length_item),0]
        ann = wfdb.rdann(local_path + file[-11:], 'atr', return_label_elements='label_store') # read annotations
        beats = np.zeros(np.shape(ecg))
        labels_interest = [1]# ,2,3,4,6,7,9,10,11,12,13,34,35,38] # annotations we take as valid beats
        beat_labels = np.all(np.stack([[True if v in labels_interest else False for v in ann.label_store],
            #                            ann.label_store>=1, # filter for labels of beats
        #                                ann.label_store<=13,
        #                                ann.label_store==34,
        #                                ann.label_store==35,
        #                                ann.label_store==38,
        #                                ann.label_store!=5,
        #                                ann.label_store!=8,
                                       1000<ann.sample, # cut off beats before intervall of interest
                                       ann.sample<1500+int(length_item) # cut off beats after intervall of interest
                                       ]), axis=0)
        beat_sample = ann.sample[beat_labels]-1000 # positions of beats in samples
        BBI = (beat_sample[1:] - beat_sample[:-1])/250*1000 # construct BBI in ms
        # print(len(BBI))
        # print([BBI[n+1] for n in range(len(BBI[1:])) if abs(BBI[n+1] - BBI[n])/BBI[n]<0.1])
        # BBI_10 = [BBI[n+1] for n in range(len(BBI[1:])) if abs(BBI[n+1] - BBI[n])/BBI[n]<0.1] # 10%-Prozent Filter
        # print(np.mean(BBI))
        BBI_10 = [BBI[n] for n in range(len(BBI)) if np.abs(np.median(BBI) - BBI[n])/np.median(BBI)<0.1] # 10%-Prozent Filter
        beat_sample_10 = [] # 10%-Prozent Filter
        for n in range(len(BBI)):
            if np.abs(np.median(BBI) - BBI[n])/np.median(BBI)<0.1:
                beat_sample_10.append(beat_sample[n+1])
            else:
                # beat_sample_10.append(int((beat_sample[n+1]-beat_sample[n])/2+beat_sample[n]))
                beat_sample_10.append(int(np.sum(BBI[:n])/1000*250*n))
        
        filter_10 = [True for n in range(len(BBI)) if np.abs(np.median(BBI) - BBI[n])/np.median(BBI)>0.1] # 10%-Prozent Filter
        # print(BBI_10)
        # print(len(BBI_10))
        # print(np.mean(BBI_10))
        # BBI_10_10 = [BBI_10[n] for n in range(len(BBI_10)) if abs(np.mean(BBI_10) - BBI_10[n])/BBI_10[n]<0.1] # 10%-Prozent Filter
        beats[beat_sample] = int(3) # timeseries of beats marked as 3 in samples with 250 Hz. Constr_feat 'Tacho' case looks for 3
        if len(BBI) > 0.75*length_item/250:# check if there are normal beats at least every two seconds
            if len(filter_10)>0: # check if filter found something
                train_data.append(ecg)
                train_beats.append(beats)
                train_BBI.append(BBI)
                print(len(train_data))
                print(BBI)
                #print(beats)
    # wfdb.show_ann_labels() # Liste der beat Labels und deren Bedeutung
    # Upsampling to 256Hz with padding constants to temporal early side
    data = np.pad(np.array(train_data), ((0,0),(int(length_item/250*6),0)))
    peaks = np.pad(np.array(train_beats), ((0,0),(int(length_item/250*6),0)))
    
    return data, peaks, train_BBI

def load_clean_data(list_training, length_item):
    """Loads ecgs and annotation of selected segments
    Segments are downloaded beforehand
    Segments are cut to shorter length for processing efficiency
    
    Input:
        list of strings. list of selected segments
        
    Return:
        data. numpy.array of floats. ecgs
        peaks. numpy.array of int. positions of r-peaks marked by int(3)
        BBI. list of arrays. int. beat-to-beat intervalls"""
    train_data = []
    train_beats = []
    train_BBI = []
    
    print("Amount of CPU cores: ", mp.cpu_count())
    # Run function parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(snippet_search, [(file, length_item) for file in list_training])
    
    for n in range(len(results)): # loop over outputs of snippet search
        for m in range(np.shape(results[n][0])[0]):
            train_data.append(results[n][0][m])
            train_beats.append(results[n][1][m])
            train_BBI.append(results[n][2][m])
    
    print("Amount of snippets: ", len(train_data))
                #print(beats)
    # wfdb.show_ann_labels() # Liste der beat Labels und deren Bedeutung
    
    
    # Upsampling to 256Hz with resampling and interpolating ecg
    data = np.array(train_data)
    print("Shape of data in load_data: ", np.shape(data))
    ratio = 250 / 256 # quotient between both samplerates
    data_up = np.zeros((len(data[:,0]), int(length_item / ratio))) # empty array to contain ecg upsampled 
    for n in range(np.shape(data)[0]):
        data_up[n,:] = np.interp(list(range(len(data_up[n,:]))), list(range(len(data[n,:]))), data[n,:]) # position r-peaks and interpolate Tachogram. x-axis in samples    
        data_up[n,:] /= max(data_up[n,:])
    
    # Upsampling to 256Hz with doubling 6 samples every second randomly with value zero
    peaks = np.array(train_beats)
    peaks_up = np.zeros((len(data[:,0]), int(length_item / ratio))) # empty array to contain ecg upsampled
    for n in range(0, np.shape(peaks)[1]-500, 250): # loop over starting points of windows
        k=0
        insercion = np.random.randint(0, high=250, size=(6)) # points where we insert doubled values
        for m in range(250): # loop over window
            peaks_up[:,n+m+k+int(n/250*6)] = peaks[:,n+m]
            if m in insercion: # insercion
                k += 1
                peaks_up[:,n+m+k+int(n/250*6)] = 0
    
    return data_up, peaks_up, train_BBI

def snippet_search(file, length_item):
    """Searches in a ecg for clean snippets of length_item"""
    train_data = []
    train_beats = []
    train_BBI = []
    
    signals, fields = wfdb.rdsamp(local_path + file[-11:]) # read physical signals
    # cut off first two seconds and extract ecg of desired length + 2 seconds added to end
    # signals is in 250Hz
    # length_item is in 250Hz. calculate accordingly
    ecg = signals[:,0]
    ann = wfdb.rdann(local_path + file[-11:], 'atr', return_label_elements='label_store') # read annotations
    
    labels_interest = [1] # annotations we take as valid beats
    offset = 500
    while offset+500+int(length_item) < len(ecg):
        # print("Offset: ", offset+500+int(length_item))
        beat_labels = np.all(np.stack([[True if v in labels_interest else False for v in ann.label_store],
                                    offset<ann.sample, # cut off beats before intervall of interest
                                    ann.sample<offset+500+int(length_item) # cut off beats after intervall of interest
                                    ]), axis=0)
        
        beat_sample = ann.sample[beat_labels]-offset # positions of beats in samples
        BBI = (beat_sample[1:] - beat_sample[:-1])/250*1000 # construct BBI in ms
        # print(BBI)
        
        filter_10 = [True for n in range(len(BBI)) if np.abs(np.median(BBI) - BBI[n])/np.median(BBI)>0.1] # 10%-Prozent Filter
        beats = np.zeros((int(length_item+500)))
        beats[beat_sample] = int(3) # timeseries of beats marked as 3 in samples with 250 Hz. Constr_feat 'Tacho' case looks for 3
        # print("Amount of beats: ", len(BBI))
        
        if len(BBI) >= 0.5*length_item/250:# check if there are normal beats at least every two seconds
            # print("Amount of found artifacts: ", len(filter_10))
            if len(filter_10) < 1: # check how often filter found something
                train_data.append(ecg[offset:offset+500+int(length_item)]/50)
                train_beats.append(beats)
                train_BBI.append(BBI)
                offset += int(length_item)
        offset += int(10*250)
    return train_data, train_beats, train_BBI