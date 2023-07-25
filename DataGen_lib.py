import numpy as np
import train_lib_final as tl
import tensorflow.keras as K
import os
import pandas
import pickle
import json
import matplotlib.pyplot as plt
import matlab.engine
import time
import multiprocessing as mp

def preprocess(chunk_size:int, INPUT_name:dict, OUTPUT_name:dict):
    """function for preprocessing huge dataset
    it prepares features of ecg
    forms them in new chunks and saves them as npy-files or json-files

    Returns:
        chunk_size(int): number of samples in new chunks containing processed data
    """
    global data_list, config
    data_list = list(OUTPUT_name.keys()) + list(INPUT_name.keys()) # list of mentioned features
    data_list = tl.unique(data_list)
    # Load config
    with open('data/CVP-dataset/dataset/CONFIG', 'rb') as fp:
        config = json.load(fp)
    # with open('data/CVP-dataset/dataset/CONFIG', 'r') as file:
    #     str_config = file.read()
    #     str_config = str_config.replace("true", "True") # Fixing boolean value
    #     str_config = str_config.replace("false", "False") # Fixing boolean value
    #     exec("config=" + str_config)
    dic_samplerate = {"RX": 256, # dictionary with samplerates of studies
                      "CVD": 256,
                      "IH": 250}
    global samplerate, chunk_path, new_chunk_path, segment_count
    samplerate = dic_samplerate[config["study_id"]]
    chunk_path = {"Training":'data/CVP-dataset/dataset/', 
                  "Test":'data/CVP-dataset/dataset/evaluation/'}
    new_chunk_path = {"Training":'data/current-set/', 
                     "Test":'data/current-set/evaluation/'}
    segment_count = {"Training":int(config["segment_count"] * 0.8), 
                     "Test":int(config["segment_count"] * 0.2)}
    print(segment_count)
    tic = time.time()
    # loop training and test
    for ToT in ["Training", "Test"]:
        new_chunk_ID = 0 # ID for new chunks
        
        # Find first and last chunk of dataset
        chunk_list = [f for f in os.listdir(chunk_path[ToT]) if os.path.isfile(os.path.join(chunk_path[ToT], f))]
        chunk_list = [int(f[:-4]) for f in chunk_list if "npy" in f] # list of numbering of chunks
        chunk_list.sort()
        chunk_start = min(chunk_list) # we save lowest as starting point
        chunk_ID = chunk_start
        chunk_end = max(chunk_list) # se save highest as ending point
        
        # create array of segment indices and containing chunk
        print("Mapping Segment to containing chunk ...")
        seg_chunk = np.zeros((segment_count[ToT],2), dtype=int)
        # seg_chunk[:,0] = np.arange(segment_count[ToT], dtype=int)
        segment_counter = 0
        for n in chunk_list:
            chunk = np.load(chunk_path[ToT] + str(n) + '.npy', allow_pickle=True)
            seg_chunk[segment_counter:segment_counter + len(chunk[:,0]), 0] = np.arange(len(chunk[:,0]), dtype=int)
            seg_chunk[segment_counter:segment_counter + len(chunk[:,0]), 1] = n
            segment_counter += len(chunk[:,0])
        seg_chunk = np.reshape(seg_chunk, (int(segment_count[ToT]/chunk_size),chunk_size,2))
        # chunk_map = seg_chunk = np.zeros((int(segment_count[ToT]/chunk_size),chunk_size,2), dtype=int)
        print(np.shape(seg_chunk))
        
        print("Preprocessing batches ...")
        # print("number of CPUs ", mp.cpu_count())
        # with mp.Pool(mp.cpu_count()) as pool:
        #     results = pool.map(load, [(seg_chunk[n,:,:], 
        #                                n, 
        #                                ToT, 
        #                                chunk_size,
        #                                INPUT_name,
        #                                OUTPUT_name) for n in range(np.shape(seg_chunk)[0])])
        # print("Sekunden für Batch-Processing", np.round(time.time()-tic,3))
        for n in range(np.shape(seg_chunk)[0]):
            load((seg_chunk[n,:,:], 
                    n, 
                    ToT, 
                    chunk_size,
                    INPUT_name,
                    OUTPUT_name))
            print("Sekunden für Batch-Processing", np.round(time.time()-tic,3))
            tic = time.time()
            
        """# Load first chunk
        chunk = np.load(chunk_path[ToT] + str(chunk_ID) + '.npy', allow_pickle=True)
        
        dataset_chunk = pandas.DataFrame( # new chunk to contain processed and batched data
            columns=["ecg", "r_pos", "snr", "subject_id", "tacho"])
        
        segment_ID = int(0)
        for i in range(segment_count[ToT]):
            if segment_ID == len(chunk[:,0]): # check if end of chunk already reached. If yes go to next chunk
                segment_ID = int(0)
                chunk_ID += 1
                print("ID des Source chunks", chunk_ID)
                if chunk_ID > chunk_end: # check if last chunk already reached. If yes break loop
                    break
                chunk = np.load(chunk_path[ToT] + str(chunk_ID) + '.npy', allow_pickle=True) # load next chunk
            
            dataset_chunk.loc[len(dataset_chunk)] = chunk[segment_ID,:]
            segment_ID += 1
            # print("Länge des dataset_chunk", len(dataset_chunk))
            if len(dataset_chunk)>=chunk_size: # check if batch size is reached
                data = dataset_chunk.to_numpy()
                
                # Seperate Features into dictionary values
                data_dic = feat_to_dic(data, data_list)
                
                X, y_list, out_types = set_items(data_dic, INPUT_name, OUTPUT_name, config["segment_length"]*samplerate)
                
                np.save( # save preprocessed and batched input data in chunk as numpy array
                    new_chunk_path[ToT] + f"X-{new_chunk_ID}", X, allow_pickle=True)
                # save preprocessed and batched output data in chunk as list
                with open(new_chunk_path[ToT] + f"y-{new_chunk_ID}", 'wb') as fp:
                    pickle.dump(y_list, fp)
                new_chunk_ID += 1
                dataset_chunk = pandas.DataFrame( # new chunk to contain processed and batched data
                                                 columns=["ecg", "r_pos", "snr", "subject_id", "tacho"])
                print("Sekunden für ein Batch", np.round(time.time()-tic,3))"""
        
def load(seg_n):
    """loads batch, preprocess and saves in new chunk

    Args:
        seg_n (_type_): _description_
    """
    seg_chunk = seg_n[0]
    new_chunk_ID = seg_n[1]
    ToT = seg_n[2]
    chunk_size = seg_n[3]
    INPUT_name = seg_n[4]
    OUTPUT_name = seg_n[5]
    # new chunk to contain processed and batched data
    dataset_chunk = pandas.DataFrame(columns=["ecg", "r_pos", "snr", "subject_id", "tacho"])
    # Load first chunk
    if len(np.unique(seg_chunk[:,1]))==1:
        chunk = np.load(chunk_path[ToT] + str(seg_chunk[0,1]) + '.npy', allow_pickle=True)
        # print(seg_chunk)
        try:
            for row in range(len(seg_chunk[:,0])):
                dataset_chunk.loc[len(dataset_chunk)] = chunk[row,:]
        except:
            print("inserting did not work")
            print(seg_chunk)
    elif len(np.unique(seg_chunk[:,1]))>1:
        for unique_chunk in np.unique(seg_chunk[:,1]):
            chunk = np.load(chunk_path[ToT] + str(unique_chunk) + '.npy', allow_pickle=True)
            try:
                for row in range(len(seg_chunk[unique_chunk==seg_chunk[:,1],0])):
                    dataset_chunk.loc[len(dataset_chunk)] = chunk[row,:]
            except:
                print("inserting did not work")
                print(seg_chunk)
        
    data = dataset_chunk.to_numpy()
    data_dic = feat_to_dic(data, data_list)
    X, y_list, out_types = set_items(data_dic, INPUT_name, OUTPUT_name, config["segment_length"]*samplerate)
    print("Saving new chunk ", new_chunk_ID)
    # save preprocessed and batched input data in chunk as numpy array
    np.save(new_chunk_path[ToT] + f"X-{new_chunk_ID}", X, allow_pickle=True)
    # save preprocessed and batched output data in chunk as list
    with open(new_chunk_path[ToT] + f"y-{new_chunk_ID}", 'wb') as fp:
        pickle.dump(y_list, fp)
    # print("Sekunden für ein Batch", np.round(time.time()-tic,3))
    print(new_chunk_ID)            

def feat_to_dic(data, data_list):
    """Seperates needed features of data into dictionary for set_items

    Args:
        data (numpy): contains features

    Returns:
        data_dic: dictionary with needed features
    """
    # Seperate feature in dictionary keys
    data_dic = {}
    Tacho_ms_check = True # check if Tachogram for Symbols and Parameters was ever loaded
    data_dic['subject_id'] = np.array(list(data[:,3]))
    for name in list(data_list):
        if name in ['SNR']:
            data_dic['SNR'] = np.array(list(data[:,2]))
        if name in ['ECG']:
            data_dic['ECG'] = np.array(list(data[:,0]))
        if name == 'Tacho':
            # beats = np.array(list(data_training[:,1]))
            # beats[beats] = int(3)
            data_dic['Tacho'] = np.array(list(data[:,4]))
            # print(data_dic['Tacho'][3])
            # print(data_dic['Tacho'][4])
            # print(data_dic['Tacho'][5])
            # exit()
        if name in ["symbolsC", "Shannon", "Polvar10", "forbword"]: # we construct non-linear parameters from BBI
            if Tacho_ms_check:
                data_dic["symbolsC"] = np.array(list(data[:,4]))*1000  # loads pointer to dataset into variable
                Tacho_ms_check = False
    return data_dic

def set_items(data, INPUT_name, OUTPUT_name, length_item: int, from_gen=False):
    """This function pulls items from datasets and separates them into input and output for training and test sets.
     
    Args:
        data (dict): Dictionary containing datasets.
        INPUT_name (dict): Selection of input features and lags.
        OUTPUT_name (dict): Selection of output features and lags.
        length_item (int): Length of items given into the neural network.
    
    Returns:
        tuple: A tuple containing numpy arrays of the training or test sets for the input and output.
    """  
    # Calculation of the features and slicing of time series
    # print("Constructing input array ...")
    X = constr_feat(data, INPUT_name, length_item, from_gen=from_gen)
    X = X[list(X.keys())[0]] # Takes value out of only key of dictionary
    # print(np.shape(X))
    # print("Constructing output array ...")
    y = constr_feat(data, OUTPUT_name, length_item, from_gen=from_gen)
    out_types = output_type(OUTPUT_name) # global list of types of output for loss and output layer
    y_list = [] # Sorting values of dictionary into list. Outputs need to be given to NN in list. Allows different formatted Truths
    for key in y.keys():
        # print("Shape of Output data for feature ", key, ": ", np.shape(y[key]))
        y_list.append(y[key])
    return X, y_list, out_types

def constr_feat(data, NAME, length_item, from_gen=False):
    """ This function constructs the feature from given datasets and lags of interest
    If given a timeseries it transforms it to a sequence with lag
    If given parameter it transform into sequence with constant value
    
    :param data: dictionary containing dataset
    :param NAME: list of features. list of strings
    :param length_item: length of items given into the neural network
    :return sequence: numpy arrays of features. rows timeseries. Columns features
     """
    # if from_gen == True:
    #     samplerate = 256
    # print(NAME)
    dic_seq = {}
    global symbols_size # saves size of BBI arrays globally. Is needed in NN construction
    if not('symbols_size' in globals()): # check if BBI_size exists globally
        symbols_size = 0
    feat_number = int(-1) # keep count of columns with features
    lag_current = 'lag -100000' # needed for symbols and words efficiency
    for key in NAME: # loop over dictionary keys
        for lag in NAME[key]: # loop over lags and options found under key
            feat_number += 1
            # check if data is timeseries, distribution or parameter            
            if key == 'ECG': # check if feature is ECG timeseries
                seq = data[key][:, int(lag[4:]) : length_item + int(lag[4:])]
                # seq *= 100 # min-max scaling by hand
                dic_seq[key+lag] = seq               
                continue
            
            if key == 'Tacho':
                # Tachogram of ecg. x-axis: sample. y-axis: ms
                # takes peak and wave categories of matlab syntethized with samplerate 1024Hz
                # identifies r-peaks and calculates BBI (distances in samples)
                # constructs Tachogram and transforms units into ms
                
                if np.shape(data[key])[1] / (length_item / samplerate) <= 8: # Check ob Datenquelle weniger als 8Hz aufweist. Ja: Direkt Tacho. Nein: Peak-Position
                    # print(np.shape(data[key])[1]) # Anzahl samples in Tacho Source Datei
                    # print(length_item) # length_item in samples
                    
                    ds_samplerate = int(2**2) # int(2**7) # Ziel samplerate beim Downsampling
                    ratio = 4 / ds_samplerate # quotient between both samplerates
                    # Tacho = np.zeros((len(data[key][:,0]), int(length_item / ratio))) # empty array to contain Tachogram
                    # for n in range(len(data[key][:,0])): # loop over all examples
                    #     Tacho[n,:] = np.interp(list(range(len(Tacho[0,:]))), list(range(len(data[key][n,:]))), data[key][n,:]) # position r-peaks and interpolate Tachogram. x-axis in samples
                    dic_seq[key+lag] = data[key][:,::int(ratio)]
                    # print(np.shape(dic_seq[key+lag]))
                    continue
                
                bc = data[key][:, int(lag[4:]) : length_item + int(lag[4:])]
                bc = bc==3 # binary categorizing of r-peaks
                rp = np.argwhere(bc>0) # position of r-peaks in samples of all examples
                ds_samplerate = int(2**1) # int(2**7) # Ziel samplerate beim Downsampling
                ratio = samplerate / ds_samplerate # quotient between both samplerates
                Tacho = np.zeros((len(bc[:,0]), int(length_item / ratio))) # empty array to contain Tachogram
                for n in range(len(bc[:,0])): # loop over all examples
                    ts =  np.argwhere(rp[:,0]==n)[:,0] # position of r-peaks of example n in rp
                    rp_ts = rp[ts,1] # position of r-peaks in example n
                    y_ts = rp_ts[1:] - rp_ts[:-1] # calculate BBI between r-peaks. Exlude first point
                    Tacho[n,:] = np.interp(list(range(len(Tacho[0,:]))), rp_ts[:-1] / ratio, y_ts) # position r-peaks and interpolate Tachogram. x-axis in samples
                    # for p in range(len(Tacho[n,:])):
                    #     Tacho[n,p] = Tacho[n,p] if abs(np.median(Tacho[n,:]) - Tacho[n,p])/np.median(Tacho[n,:])<0.1 else Tacho[n,p-1]
                        # BBI_10[n] for n in range(len(BBI_10)) if abs(np.mean(BBI_10) - BBI_10[n])/BBI_10[n]<0.1
                Tacho = Tacho / samplerate # transform from sample into ms
                dic_seq[key+lag] = Tacho
                continue
            
            if 'symbols' in key: # Transform into sparse categorical vector
                
                # symbols and words are needed often
                # calculate and reuse if lag same
                # if key in ['symbols', 'words', 'parameters', 'parametersTacho', 'parametersSymbols', 'parametersWords']:
                if int(lag[4:])!=int(lag_current[4:]):
                    lag_current = lag
                    symbols, words, BBI_list = cut_BBI(data[key], lag_current, length_item)
                
                # Upsampling to length_item / ds_ratio
                ds_samplerate = int(2**2) # Ziel samplerate beim Downsampling
                ds = int(samplerate/ds_samplerate) # downsampling ratio
                sym_up = np.full((len(symbols), int(length_item/ds)), int(10)) # array of 10s. After Upsampling no 10s expected
                for example in range(len(BBI_list)): # extract BBI of example in ms
                    BBI = np.array(BBI_list[example] / 1000 * ds_samplerate) # transform values from ms into sample (of down_samplerate)
                    BBI = np.cumsum(BBI).astype(int) # cummulation BBI. this way we get point in time for each beat
                    u, counts = np.unique(BBI, return_counts=True) # checks if two beats are on one sample
                    if max(counts) > 1:
                        print("Warning: Downsamplerate of Symbols is too low. Two beats on one sample in example:", example)
                    sym_up[example, :BBI[0]] = symbols[example][0] # values before first BBI
                    for n in range(1,len(BBI)): # loop over single points of BBI and symbols
                        sym_up[example, BBI[n-1]:BBI[n]] = symbols[example][n] # BBI set for samples before r-peak
                    sym_up[example, BBI[-1]:] = symbols[example][-1] # values after last BBI
                
                
                dic_seq[key+lag] = sym_up.astype(np.int32)
                continue
            
            if 'Shannon' == key: # calculate Shannon entropy
                
                # symbols and words are needed often
                # calculate and reuse if lag same
                if int(lag[4:])!=int(lag_current[4:]):
                    lag_current = lag
                    symbols, words, BBI_list = cut_BBI(data['symbolsC'], lag_current, length_item)
                
                future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
                engine = future.result() # run engine in background
                engine.cd(r'nl', nargout=0) # change directory to nl folder
            
                words = words.astype(np.float64)
                
                # fwshannon = lambda distribution: engine.fwshannon(distribution, nargout=1)
                # with mp.Pool(mp.cpu_count()) as pool:
                #     results = pool.map(fwshannon, [distribution[0,:] for distribution in np.split(words,np.shape(words)[0],axis=0)])
                fwshannon_param = []
                for row in [distribution[0,:] for distribution in np.split(words,np.shape(words)[0],axis=0)]: # loop over examples
                    fwshannon_param.append(engine.fwshannon(row, nargout=1)) # Very important matlab functions defined for float64 / double
                fwshannon_param = np.array(fwshannon_param, dtype=np.float16) / 4
                # print(np.max(fwshannon_param))
                # print(np.min(fwshannon_param))                
                dic_seq[key+lag] = fwshannon_param
                continue
            
            if 'Polvar10' == key: # calculate Shannon entropy
                
                # symbols and words are needed often
                # calculate and reuse if lag same
                if int(lag[4:])!=int(lag_current[4:]):
                    lag_current = lag
                    symbols, words, BBI_list = cut_BBI(data['symbolsC'], lag_current, length_item)
                
                future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
                engine = future.result() # run engine in background
                engine.cd(r'nl', nargout=0) # change directory to nl folder
            
                # words = words.astype(np.float64)
                
                # fwshannon = lambda distribution: engine.fwshannon(distribution, nargout=1)
                # with mp.Pool(mp.cpu_count()) as pool:
                #     results = pool.map(fwshannon, [distribution[0,:] for distribution in np.split(words,np.shape(words)[0],axis=0)])
                plvar_10_param = []
                for BBI in BBI_list: # loop over examples
                    plvar_10_param.append(engine.plvar(BBI, 10, nargout=1)) # Very important matlab functions defined for float64 / double
                plvar_10_param = np.array(plvar_10_param, dtype=np.float16) + 0.1
                dic_seq[key+lag] = plvar_10_param
                continue
            
            if 'forbword' == key: # calculate Shannon entropy
                
                # symbols and words are needed often
                # calculate and reuse if lag same
                if int(lag[4:])!=int(lag_current[4:]):
                    lag_current = lag
                    symbols, words, BBI_list = cut_BBI(data['symbolsC'], lag_current, length_item)
                
                future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
                engine = future.result() # run engine in background
                engine.cd(r'nl', nargout=0) # change directory to nl folder
            
                words = words.astype(np.float64)
                
                # fwshannon = lambda distribution: engine.fwshannon(distribution, nargout=1)
                # with mp.Pool(mp.cpu_count()) as pool:
                #     results = pool.map(fwshannon, [distribution[0,:] for distribution in np.split(words,np.shape(words)[0],axis=0)])
                forbword = []
                for row in [distribution[0,:] for distribution in np.split(words,np.shape(words)[0],axis=0)]: # loop over examples
                    forbword.append(engine.forbidden_words(row, nargout=1)) # Very important matlab functions defined for float64 / double
                forbword = np.array(forbword, dtype=np.float16) / 40 + 0.1
                # plvar_10_param = np.array(plvar_10_param, dtype=np.float16) / np.max(plvar_10_param)
                dic_seq[key+lag] = forbword
                continue
            
            if 'SNR' in key: # calculate SNR of ECG 
                try: 
                    seq = data['SNR'][:, :256]
                except:
                    seq = data['ECG'][:, int(lag[4:]) : length_item + int(lag[4:])]
                snr = []
                for k in range(len(seq[:,0])):
                    snr.append(signaltonoise(seq[k,:]))
                snr = np.array(snr)
                dic_seq[key+lag] = snr             
    return dic_seq

def output_type(OUTPUT_name):
    """ returns list with type of output at the corresponding column
    
    returns: list. Types of output for each column
    """
    out_types = []
    dic_types = {
                    "ECG": "regressionECG",
                    "Tacho": "regressionTacho",
                    "symbolsC": "classificationSymbols", 
                    "Shannon": "Shannon", 
                    "Polvar10": "Polvar10", 
                    "forbword": "forbword",
                    "SNR": "regressionSNR",
                }
    for key in OUTPUT_name: # loop over dictionary keys
        for name in OUTPUT_name[key]: # loop over lags found under key
            out_types.append(dic_types[key])
    return out_types

def cut_BBI(data, lag, length_item):
    """Function to cut interval of interest from BBI timeseries and return symbols and words
    """
    # wir nutzen Funktionen geschrieben in Matlab
    # Gecodet von matthias
    
    # Cutting BBI fitting our needs
    # lower bound / starting point of BBI time series. Defined by lag in samples / data points
    BBI_list = [] # saving in a list allows different length of sequences
    if isinstance(data, list): # check if data is a list. numpy.array otherwise
        for d in data: # loop over each BBI timeseries
            lb_BBI = np.where(np.cumsum(d) >= float(lag[4:])/samplerate*1000, d, 0)
            # upper bound / ending point of BBI
            up_BBI = np.where(np.cumsum(lb_BBI) <= length_item/samplerate*1000, lb_BBI, 0)
            BBI_list.append(up_BBI[up_BBI!=0]) # save only non-zero elements
            del lb_BBI, up_BBI
    else:
        lb_BBI = np.where(np.cumsum(data, axis=1) >= float(lag[4:])/samplerate*1000, data, 0)
        # upper bound / ending point of BBI
        up_BBI = np.where(np.cumsum(lb_BBI, axis=1) <= length_item/samplerate*1000, lb_BBI, 0)
        BBI = up_BBI[:, ~np.all(up_BBI == 0, axis = 0)] # cut all columns with only zeros out
        for n in range(len(BBI[:,0])):
            bbi = BBI[n,:]
            BBI_list.append(bbi[~(bbi==0)]) # BBI is in ms
    # Extracting symbols from cut BBI
    symbols, words = calc_symboldynamics(BBI_list) # Outputs lists of arrays. Symbols have different length
    return symbols, words, BBI_list

def signaltonoise(a, axis=0, ddof=0):
    """Calculates SNR for 1s windows
    Sequence is reshaped into 2d array with dimensions (samplerate, length in seconds)    
    
    Args:
        a (double): ecg sequence
        axis (int, optional): _description_. Defaults to 0.
        ddof (int, optional): _description_. Defaults to 0.

    Returns:
        double: sequence of SNR of 1s windows. 1D with length of measurements in seconds
    """
    a = np.reshape(a, (samplerate,-1))
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    
    return np.where(sd == 0, 0, m/sd)

def calc_symboldynamics(BBI): #beat_to_beat_intervals, a, mode
    """ Function to determine symbols and words for dynamics of beat-to-beat-intervals
    
    
    returns:    symbols. list of arrays with different length. Contains Categories of BBI dynamics
                words. distribution of three letter words consisting of 4-category symbols
    """
    future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
    engine = future.result() # run engine in background
    engine.cd(r'nl', nargout=0) # change directory to nl folder
    symbols = [] # list of symbol sequences
    words = [] # list of word distributions
    for row in range(len(BBI[:])): # loop over examples
        # insert one-dim array from list into matlab function
        # function returns us symbols and words as output
        result = engine.calc_symboldynamics(BBI[row], 0.01, "movdiff", nargout=2) # vielleicht mit alpha 0.05 probieren
        flatten = [result[1][n][0] for n in range(len(result[1]))] # flatten the list of matlab.doubles
        # print(np.array(flatten,dtype=int))
        symbols.append(np.array(flatten,dtype=int))
        words.append(result[0])
        
        # Here we will add non-linear parameters calculated from symbols and words
        
    words = np.array(words, dtype=int)[:,:,0] # transform list of distributions into numpy array for faster processing
    engine.quit()
    
    return symbols, words

class DataGenerator(K.utils.Sequence):
    "Generates data for Keras of CVP dataset"
    def __init__(self, chunk_size, INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, batch_size=32, ToT="Training"): # , shuffle=False
        "Initialization"
        self.INPUT_name = INPUT_name
        self.OUTPUT_name = OUTPUT_name
        self.data_list = tl.unique(list(OUTPUT_name.keys()) + list(INPUT_name.keys())) # list of mentioned features
        self.batch_size = batch_size
        self.chunk_size = chunk_size # size of chunks containing preprocessed data
        if chunk_size % batch_size != 0:
            print("chunk ", chunk_size," and batch size ", batch_size," must have modulo 0")
        # self.shuffle = shuffle
        # self.list_IDs = list_IDs
        # self.data_X = X
        # self.data_y = y
        "Loading CONFIG-file"
        # config = load('data/CVP-dataset/dataset/CONFIG')
        with open('data/CVP-dataset/dataset/CONFIG', 'rb') as fp:
            self.config = json.load(fp)
        self.length_item = int(self.config["segment_length"]) # length of ecg inputs in s
        # self.indexes = np.arange(int(self.config["segment_count"]))
        if self.ToT == "Training":
            return int(np.floor(self.config["segment_count"]*0.8/self.batch_size))
        else:
            return int(np.floor(self.config["segment_count"]*0.2/self.batch_size))
        self.ToT = ToT
        if ToT == "Training":
            self.chunk_path = 'data/current-set/'
        else:
            self.chunk_path = 'data/current-set/evaluation/'
        chunk_list = [f for f in os.listdir(self.chunk_path) if os.path.isfile(os.path.join(self.chunk_path, f))]
        chunk_list = [int(f[:-4]) for f in chunk_list if "npy" in f] # list of numbering of chunks
        self.chunk_start = min(chunk_list) # we save lowest as starting point
        self.chunk_ID = self.chunk_start
        self.chunk_end = max(chunk_list) # se save highest as ending point
        self.segment_ID = int(0)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.data_X[:,0]) / self.batch_size))
        if self.ToT == "Training":
            return int(np.floor(self.config["segment_count"]*0.8/self.batch_size))
        else:
            return int(np.floor(self.config["segment_count"]*0.2/self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index)
        
        return (X, y)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(int(self.config["segment_count"]))
        # self.segment_ID = int(0)
        self.chunk_ID = 0
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
    
    def __data_generation(self, index):
        """Loads batches from chunks

        Args:
            chunk_ID (int): ID of needed chunk
        """
        X = np.load(self.chunk_path + 'X-' + str(index) + '.npy', allow_pickle=True)
        with open(self.chunk_path + 'y-' + str(index), 'rb') as fp:
            y = pickle.load(fp)
        return X, y
    
    # def __data_generation_old(self, indexes):
    #     """Veraltet war zu langsame Variante"""
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Load examples from chunks
    #     chunk = np.load(self.chunk_path + str(self.chunk_ID) + '.npy', allow_pickle=True)
        
    #     dataset_chunk = pandas.DataFrame(
    #         columns=["ecg", "r_pos", "snr", "subject_id", "tacho"])
        
    #     for i, ID in enumerate(indexes):
    #         if self.segment_ID == len(chunk[:,0]): # check if end of chunk already reached. If yes go to next chunk
    #             self.segment_ID = 0
    #             self.chunk_ID += 1
    #             if self.chunk_ID > self.chunk_end: # check if last chunk already reached. If yes break loop
    #                 break
    #             chunk = np.load(self.chunk_path + str(self.chunk_ID) + '.npy', allow_pickle=True)
    #         # data.append(chunk[self.segment_ID,:])
    #         dataset_chunk.loc[len(dataset_chunk)] = chunk[self.segment_ID,:]
    #         self.segment_ID += 1
    #     data = dataset_chunk.to_numpy()
    #     # print(data[0,:])
        
    #     # Seperate feature in dictionary keys
    #     data_dic = {}
    #     Tacho_ms_check = True # check if Tachogram for Symbols and Parameters was ever loaded
    #     data_dic['subject_id'] = np.array(list(data[:,3]))
    #     for name in list(self.data_list):
    #         if name in ['SNR']:
    #             data_dic['SNR'] = np.array(list(data[:,2]))
    #         if name in ['ECG']:
    #             data_dic['ECG'] = np.array(list(data[:,0]))
    #         if name == 'Tacho':
    #             # beats = np.array(list(data_training[:,1]))
    #             # beats[beats] = int(3)
    #             data_dic['Tacho'] = np.array(list(data[:,4]))
    #             # print(data_dic['Tacho'][3])
    #             # print(data_dic['Tacho'][4])
    #             # print(data_dic['Tacho'][5])
    #             # exit()
    #         if name in ["symbolsC", "Shannon", "Polvar10", "forbword"]: # we construct non-linear parameters from BBI
    #             if Tacho_ms_check:
    #                 data_dic["symbolsC"] = np.array(list(data[:,4]))*1000  # loads pointer to dataset into variable
    #                 Tacho_ms_check = False
                    
    #     # Generate data   
    #     X, y_list, out_types = tl.set_items(data_dic, self.INPUT_name, self.OUTPUT_name, self.length_item*256)

    #     return X, y_list