# This library if for processing an ECG to BBI, words, symbols and non-linear parameters

import os
from random import sample
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Dense,
    Input,
    LSTM,
    Conv1D,
    UpSampling1D,
    AveragePooling1D,
    MaxPooling1D,
    Flatten,
    concatenate,
)
from keras.models import Model
import matlab.engine


# print("Tensorflow version: ", tf.__version__)
# exit()

# Change directory to current file directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def unique(list1):
    """Sorts out multiple occuring elements in list

    Args:
        list1 (list): list of datasets we are interested in

    Returns:
        list: unique list of dataset names
    """
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
        
def memorize(filename, dataset):
    """Memorizes needed data from HDF5 file and returns as a dictionary.
    
    Args:
        filename (str): Name of the HDF5 file.
        dataset (list): List of timeseries and parameters to be retrieved from the HDF5 file.
    
    Returns:
        tuple: A tuple containing the data as a dictionary and the sample rate.
    """
    # Check if filename is a string
    if not isinstance(filename, str):
        raise TypeError("filename must be a string naming a H5-file.")
    # Check if dataset is a list
    if not isinstance(dataset, list):
        raise TypeError("dataset must be a list of keys as datasets of the H5-file.")
    
    # Try to open the HDF5 file
    try:
        file = h5py.File(filename, "r")  # pointer to h5-file
    except:
        print("Current directory:", dname)
        dir_list=[]
        dir_list.append([".data/" + name for name in os.listdir("./data") if "h5" in name]) # listet alle h5 Dateien im data Ordner
        dir_list.append([".data/" + name for name in os.listdir("./data") if "hdf5" in name])
        dir_list.append([".data/" + name for name in os.listdir() if "h5" in name]) # listet alle h5 Dateien im aktuellen Ordner
        dir_list.append([".data/" + name for name in os.listdir() if "hdf5" in name])
        raise FileNotFoundError("H5-file not found in current directory. Choose one of the files below: {dir_list}")
    
    data = {} # dictionary containing data
    for name in dataset:  # loop over datasets named in list
        print("Memorizing timeseries: ", name)
        if name in ["symbolsC", "words", 
                    "parameters", "parametersTacho", "parametersSymbols", "parametersWords"]: # we construct non-linear parameters from BBI
                data[name] = file["BBI"][:]  # loads pointer to dataset into variable
                print("Shape of " + name, np.shape(data[name]), " source data")
        else:
            try:
                data[name] = file[name][:]  # loads pointer to dataset into variable
                print("Shape of " + name, np.shape(data[name]))
            except KeyError:
                if name == "Tacho":  # we will use Tachogramm with x-axis in samples
                    data[name] = file["RP"][:]  # loads pointer to dataset into variable
                    print("Shape of " + name, np.shape(data[name]))
                else:
                    raise KeyError(f"dataset {name} not found in H5-file. Choose one of the datasets below: {file.keys()}")
    
    # plots the distribution of first dataset
    plt.figure(1)
    plt.title("Distribution of data " + dataset[0])
    data_flat = data[dataset[0]].flatten()
    plt.hist(data_flat, bins=30)
    plt.xlabel("time in ms")
    plt.ylabel("occurence")
    plt.savefig("distribution of data")
    plt.close()
    
    global samplerate # extract the samplerate of ecg timeseries
    samplerate = file.attrs['samplerate'][0]
    
    return data, samplerate # returns datasets

def set_items(data, INPUT_name, OUTPUT_name, length_item):
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
    print("Constructing input array ...")
    X = constr_feat(data, INPUT_name, length_item)
    X = X[list(X.keys())[0]] # Takes value out of only key of dictionary
    print(np.shape(X))
    print("Constructing output array ...")
    y = constr_feat(data, OUTPUT_name, length_item)
    out_types = output_type(y, OUTPUT_name) # global list of types of output for loss and output layer
    y_list = [] # Sorting values of dictionary into list. Outputs need to be given to NN in list. Allows different formatted Truths
    for key in y.keys():
        y_list.append(y[key])
    return X, y_list, out_types

def output_type(dic_seq, OUTPUT_name):
    """ returns list with type of output at the corresponding column
    Types are of regressional and categorical timeseries and parameters
    The three types need to be handled differently in the output layer and loss function to get the desired results
    
    returns: list. Types of output for each column
    """
    dic_types = {
        "ECG": "regressionECG",
        "Tacho": "regressionTacho",
        "symbolsC": "classificationSymbols", 
        "words": "distributionWords",
        "parameters": "parameter", 
        "parametersTacho": "parameterTacho", "parametersSymbols": "parameterSymbols", "parametersWords": "parameterWords",
    }
    # global because needed in loss function. Just easier to implement like this
    global out_types
    out_types = []
    for key in OUTPUT_name: # loop over dictionary keys
        for name in OUTPUT_name[key]: # loop over lags found under key
            try:
                # if key == 'RP': # Check ob Symbole einbezogen werden
                #     out_types.append(dic_types[key] + str(0)) # number columns containing label
                #     # Important for initializing model, loss function and plotting output
                if key == 'symbolsC': # Check ob Symbole einbezogen werden mit Klassifikation
                    out_types.append(dic_types[key] + str(len(np.unique(dic_seq[key+name])))) # number columns containing label
                        # Important for initializing model, loss function and plotting output
                elif key == 'words':
                    out_types.append(dic_types[key] + str(len(dic_seq[key+name][0]))) # number columns containing counts of words
                    # for k in range(len(dic_seq[key+name][0])): # unique labels in timeseries
                    #     out_types.append(dic_types[key] + str(k)) # number columns containing label
                        # Important for initializing model, loss function and plotting output
                elif 'parameters' in key:
                    print(np.shape(dic_seq[key+name]))
                    out_types.append(dic_types[key] + str(len(dic_seq[key+name][0]))) # number columns containing parameters
                else:
                    out_types.append(dic_types[key]) # regression timeseries
            except:
                out_types.append("No type detected")
    return out_types

def constr_feat(data, NAME, length_item):
    """ This function constructs the feature from given datasets and lags of interest
    If given a timeseries it transforms it to a sequence with lag
    If given parameter it transform into sequence with constant value
    
    :param data: dictionary containing dataset
    :param NAME: list of features. list of strings
    :param length_item: length of items given into the neural network
    :return sequence: numpy arrays of features. rows timeseries. Columns features
     """
    print(NAME)
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
            print("Shape of data ", key, " : ", np.shape(data[key]))
            print("feature: lag ", int(lag[4:]), "at column ", feat_number) # lag is in given in samples / datapoints
            
            if key == 'ECG': # check if feature is ECG timeseries
                seq = data[key][:, int(lag[4:]) : length_item + int(lag[4:])]
                seq *= 100 # min-max scaling by hand
                dic_seq[key+lag] = seq               
                continue
            
            if key == 'Tacho':
                # Tachogram of ecg. x-axis: sample. y-axis: ms
                # takes peak and wave categories of matlab syntethized with samplerate 1024Hz
                # identifies r-peaks and calculates BBI (distances in samples)
                # constructs Tachogram and transforms units into ms
                
                bc = data[key][:, int(lag[4:]) : length_item + int(lag[4:])]
                bc = bc==3 # binary categorizing of r-peaks
                rp = np.argwhere(bc>0) # position of r-peaks in samples of all examples
                ds_samplerate = int(2**2) # int(2**7) # Ziel samplerate beim Downsampling
                ratio = samplerate / ds_samplerate # quotient between both samplerates
                Tacho = np.zeros((len(bc[:,0]), int(length_item / ratio))) # empty array to contain Tachogram
                for n in range(len(bc[:,0])): # loop over all examples
                    ts =  np.argwhere(rp[:,0]==n)[:,0] # position of r-peaks of example n in rp
                    rp_ts = rp[ts,1] # position of r-peaks in example n
                    y_ts = rp_ts[1:] - rp_ts[:-1] # calculate BBI between r-peaks. Exlude first point
                    Tacho[n,:] = np.interp(list(range(len(Tacho[0,:]))), rp_ts[1:] / ratio, y_ts) # position r-peaks and interpolate Tachogram. x-axis in samples
                Tacho = Tacho / samplerate # transform from sample into ms
                dic_seq[key+lag] = Tacho
                plt.figure(1)
                # plt.plot(list(range(len(Tacho[n,:]))), Tacho[n,:])
                plt.plot(np.linspace(0, len(Tacho[n,:]) / ds_samplerate, num=len(Tacho[n,:])), Tacho[n,:])
                plt.savefig("Tachogram.png")
                plt.close()
                continue
            
            if 'symbols' in key: # Transform into sparse categorical vector
                
                # symbols and words are needed often
                # calculate and reuse if lag same
                # if key in ['symbols', 'words', 'parameters', 'parametersTacho', 'parametersSymbols', 'parametersWords']:
                if int(lag[4:])!=int(lag_current[4:]):
                    lag_current = lag
                    symbols, words, BBI_list = cut_BBI(data[key], lag_current, length_item)
                
                # Upsampling to length_item / ds_ratio
                ds_samplerate = int(2**7) # Ziel samplerate beim Downsampling
                ds = int(samplerate/ds_samplerate) # downsampling ratio
                sym_up = np.full((len(symbols), int(length_item/ds)), int(10)) # array of 10s. After Upsampling no 10s expected
                
                for example in range(len(BBI_list)): # extract BBI of example in ms
                    BBI = np.array(BBI_list[example] / 1000 * ds_samplerate, dtype=int) # transform values into sample (of down_samplerate)
                    BBI = np.cumsum(BBI) # cummulation BBI
                    sym_up[example, :BBI[0]] = symbols[example][0] # values before first BBI
                    for n in range(len(BBI)-1): # loop over single points of BBI and symbols
                        sym_up[example, BBI[n]:BBI[n+1]] = symbols[example][n]
                    sym_up[example, BBI[n+1]:] = symbols[example][n+1] # values after last BBI
                
                
                dic_seq[key+lag] = sym_up.astype(np.int32)
                continue
                
            if key == 'words': # combine symbol dynamics into 3 letter words
                
                # symbols and words are needed often
                # calculate and reuse if lag same
                if int(lag[4:])!=int(lag_current[4:]):
                    lag_current = lag
                    symbols, words, BBI_list = cut_BBI(data[key], lag_current, length_item)
                    
                print(words)
                print(np.shape(words))
                
                dic_seq[key+lag] = words
                continue
                
            if 'parameters' in key: # calculate non-linear parameters from BBI, symbols and words
                
                # symbols and words are needed often
                # calculate and reuse if lag same
                if int(lag[4:])!=int(lag_current[4:]):
                    lag_current = lag
                    symbols, words, BBI_list = cut_BBI(data[key], lag_current, length_item)

                if key == 'parametersTacho':
                    # BBI parameters plvar and phvar
                    plvar_5_param, plvar_10_param, plvar_20_param = plvar(BBI_list)
                    phvar_20_param, phvar_50_param, phvar_100_param = phvar(BBI_list)
                    param_list = [plvar_5_param, plvar_10_param, plvar_20_param, 
                                    phvar_20_param, phvar_50_param, phvar_100_param
                                    ]
                elif key == 'parametersSymbols':
                    # symbols parameter wsdvar
                    wsdvar_param = wsdvar(symbols)
                    param_list = [wsdvar_param]
                elif key == 'parametersW':
                    # words parameters
                    forbword, fwshannon_param, fwrenyi_025_param, fwrenyi_4_param, wpsum02_param, wpsum13_param = words_parameters(words)
                    param_list = [forbword, fwshannon_param, 
                                    fwrenyi_025_param, fwrenyi_4_param, 
                                    wpsum02_param, wpsum13_param]
                else:
                    # BBI parameters plvar and phvar
                    plvar_5_param, plvar_10_param, plvar_20_param = plvar(BBI_list)
                    phvar_20_param, phvar_50_param, phvar_100_param = phvar(BBI_list)
                    
                    # symbols parameter wsdvar
                    wsdvar_param = wsdvar(symbols)
                    
                    # words parameters
                    forbword, fwshannon_param, fwrenyi_025_param, fwrenyi_4_param, wpsum02_param, wpsum13_param = words_parameters(words)
                    param_list = [plvar_5_param, plvar_10_param, plvar_20_param, 
                                    phvar_20_param, phvar_50_param, phvar_100_param,
                                    wsdvar_param,
                                    forbword, fwshannon_param, 
                                    fwrenyi_025_param, fwrenyi_4_param, 
                                    wpsum02_param, wpsum13_param]
                    
                param_arr = np.transpose(np.array(param_list, 
                                     dtype=np.float16))
                dic_seq[key+lag] = param_arr
                continue
    return dic_seq

def draw_model(model):
    """
    plots model and saves in model.ong
    """
    tf.keras.utils.plot_model(model)

def feat_check(X,y):
    """
    Plots the different features of input and output in one diagramm
    Helps with deciding if features are reasonable and correct
    """
    # print(np.shape(X))
    plt.figure(1)
    # plt.title("ECG with 5 minute duration")
    plt.plot(list(range(len(X[0,:]))), X[0,:])
    plt.savefig("Test-X.png")
    plt.close()
    
    k = 0
    for n in range(len(y)): # loop over all features by going through arrays in list
        plt.figure(2+n)
        if isinstance(y[n], list):
            data = y[n][0]
        else:
            # print(np.shape(y[n]))
            data = y[n][0,:]
        plt.plot(list(range(len(data))), data + k)
        k = k + 0.0005
        plt.savefig("Test-y-" + str(n) +".png")
        plt.close()

def setup_Conv_AE_LSTM_P(input_shape, size, samplerate):
    """This function is for testing the current preferred Architecture
    
    builds Autoencoder for different sizes and number of features
    Encoder part are convolutional layers which downsampling to half length of timeseries with every layer
    - with kernelsize corresponding to two seconds. 2s snippets contain one heartbeat for sure
    second part consists of pseudo-task branches corresponding to the selected features
    in these branches we make a prediction with the LSTM layer
    In the decoding part we use UpSampling and Conv layer to get the back to an ecg timeseries

    :param input_shape: the shape of the input array
    :param size: the width of the first encoder layer
    :param number_feat: number of features of output (number of pseudo-tasks)
    :param samplerate: samplerate of measurement. Needed for kernel size in conv layer
    :return model: keras model
    """
    
        
    # here we determine to you use mixed precision
    # Meaning that the forwardpropagation is done in float16 for higher throughput
    # And Backpropagation in float32 to keep high precision in weight adjusting
    # Warning: If custom training is used. Loss Scaling is needed. See https://www.tensorflow.org/guide/mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Input Shape:", input_shape)
    # initialize our model
    # our input layer
    Input_encoder = Input(shape=input_shape)  # np.shape(X)[1:]
    # downsampling step of 2 is recommended. This way a higher resolution is maintained in the encoder
    ds_step =  int(2**1)# factor of down- and upsampling of ecg timeseries
    ds_samplerate = int(2**7) # Ziel samplerate beim Downsampling
    orig_a_f = int(2**0) # first filter amount. low amount of filters ensures faster learning and training
    amount_filter = orig_a_f
    encoder = Conv1D(amount_filter, # number of columns in output. filters
                     samplerate*2, # kernel size. We look at 2s snippets
                     padding = "same", # only applies kernel if it fits on input. No Padding
                     dilation_rate=2, # Kernel mit regelmäßigen Lücken. Bsp. jeder zweite Punkt wird genommen
                     activation = "relu"
                     )(Input_encoder)  # downgrade numpy to v1.19.2 if Tensor / NumpyArray error here
    encoder = AveragePooling1D(ds_step)(encoder)
    print("Downsampled to: ", int(samplerate/ds_step), " Hz") # A samplerate under 100 Hz will decrease analysis quality. 10.4258/hir.2018.24.3.198
    
    # our hidden layer / encoder
    # decreasing triangle
    k = ds_step # needed to adjust Kernelsize to downsampled samplerate
    while ds_samplerate < int(samplerate/k): # start loop so long current samplerate is above goal samplerate
        if ds_samplerate > int(samplerate/k/ds_step):
            ds_step = 2
        k *= ds_step
        amount_filter *= 2
        encoder = Conv1D(amount_filter, # number of columns in output. filters
                     int(samplerate/k), # kernel size. We look at 2s snippets
                     padding = "same", # only applies kernel if it fits on input. No Padding
                     dilation_rate=2,
                     activation = "relu"
                     )(encoder)  # downgrade numpy to v1.19.2 if Tensor / NumpyArray error here
        encoder = AveragePooling1D(ds_step)(encoder)
        print("Downsampled to: ", int(samplerate/k), " Hz")
        
        
    
    # pred = Dense(size)(encoder)
    # pred = Dense(size)(pred)
    pred = LSTM(size, return_sequences=True)(encoder)
    # pred = MaxPooling1D(8)(pred)
    # pred = LSTM(size, return_sequences=True)(pred)
    # pred = MaxPooling1D(2)(pred)
    # pred = LSTM(size, return_sequences=True)(pred)
    # pred = MaxPooling1D(2)(pred)
    # pred = LSTM(size, return_sequences=True)(pred)
    # pred = LSTM(size, return_sequences=True)(pred)
    # pred = AveragePooling1D(4)(pred)
    # pred = LSTM(size, return_sequences=True)(pred)
    # encoder = Dense(size)(encoder)
    # # LSTM branch
    # lstm_br = LSTM(size, return_sequences=True)(encoder) # vielleicht amount_filter statt size
    # lstm_br = LSTM(size, return_sequences=True)(lstm_br)
    # # # Dense branch
    # dense_br = Dense(size)(encoder)
    # dense_br = Dense(size)(dense_br)
    # # concat
    # pred = concatenate([lstm_br, dense_br])
    # pred = Dense(size)(con_br)
    
        
    # branching of the pseudo-tasks
    # expanding triangle / decoder until all branches combined are as wide as the input layer
    branch_dic = {}  # dictionary for the branches
    latent_a_f = amount_filter
    for x in range(len(out_types)):
        if 'ECG' in out_types[x]:  
            amount_filter = latent_a_f      
            branch_dic["branch{0}".format(x)] = Conv1D(amount_filter,
                                                        1,
                                                        strides=1,
                                                        padding = "same")(
                                                        pred)
            branch_dic["branch{0}".format(x)] = UpSampling1D(ds_step)(branch_dic["branch{0}".format(x)])
            amount_filter /= 2
            while amount_filter >= orig_a_f:
                branch_dic["branch{0}".format(x)] = Conv1D(amount_filter,
                                                            1,
                                                            strides=1,
                                                            padding = "same")(
                                                            branch_dic["branch{0}".format(x)])
                branch_dic["branch{0}".format(x)] = UpSampling1D(ds_step)(branch_dic["branch{0}".format(x)])
                amount_filter /= 2
            branch_dic["branch{0}".format(x)] = Conv1D(1,
                                                        1,
                                                        strides=1,
                                                        padding = "same",
                                                        name = "ECG_output",
                                                        activation="linear")(
                                                        branch_dic["branch{0}".format(x)])
        
        elif 'regressionTacho' in out_types[x]: # Tachogram regression output
            # branch_dic["branch{0}".format(x)] = LSTM(size, return_sequences=True)(pred)
            branch_dic["branch{0}".format(x)] = MaxPooling1D(8)(pred)
            branch_dic["branch{0}".format(x)] = LSTM(size, return_sequences=True)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = AveragePooling1D(4)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = LSTM(size, return_sequences=True)(branch_dic["branch{0}".format(x)])
            # branch_dic["branch{0}".format(x)] = AveragePooling1D(2)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = Dense(1, activation="linear", name="Tacho_output")(branch_dic["branch{0}".format(x)])
        
        elif 'classificationSymbols' in out_types[x]: # symbols classification output
            # branch_dic["branch{0}".format(x)] = MaxPooling1D(8)(pred)
            branch_dic["branch{0}".format(x)] = LSTM(int(size), return_sequences=True)(pred)
            # branch_dic["branch{0}".format(x)] = AveragePooling1D(4)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = LSTM(int(size), return_sequences=True)(branch_dic["branch{0}".format(x)])
            # branch_dic["branch{0}".format(x)] = AveragePooling1D(4)(branch_dic["branch{0}".format(x)])
            amount_cat = extract_number(out_types[x]) # extracts number of categories from type-description
            print("Anzahl an Symbol-Kategorien: ", amount_cat)
            branch_dic["branch{0}".format(x)] = Dense(amount_cat, activation='softmax' , name='Symbols_output')(branch_dic["branch{0}".format(x)])
            
        elif 'distributionWords' in out_types[x]: # words distribution output
            branch_dic["branch{0}".format(x)] = concatenate([pred, branch_dic["branch{0}".format(x-1)]])
            
            branch_dic["branch{0}".format(x)] = LSTM(4, return_sequences=True)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = AveragePooling1D(ds_step)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = LSTM(1, return_sequences=True)(branch_dic["branch{0}".format(x)])
            
            amount_cat = extract_number(out_types[x]) # extracts number of categories from type-description
            print("Anzahl an Word-bins: ", amount_cat)
            
            branch_dic["branch{0}".format(x)] = Flatten()(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = Dense(amount_cat, activation='linear' , name='Words_output')(branch_dic["branch{0}".format(x)])
            
        elif 'parametersTacho' in out_types[x]: # parameters of BBI vector
            # branch_dic["branch{0}".format(x)] = MaxPooling1D(8)(pred)
            # branch_dic["branch{0}".format(x)] = LSTM(size, return_sequences=True)(branch_dic["branch{0}".format(x)])
            # branch_dic["branch{0}".format(x)] = AveragePooling1D(4)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = LSTM(size, return_sequences=True)(pred)
            
            amount_cat = extract_number(out_types[x]) # extracts number of categories from type-description
            print("Anzahl an parameter: ", amount_cat)
            
            branch_dic["branch{0}".format(x)] = Flatten()(branch_dic["branch{0}".format(x)])
            # Automate naming layer with out_types to ensure uniqueness
            branch_dic["branch{0}".format(x)] = Dense(amount_cat, activation='linear' , name='parameter_output')(branch_dic["branch{0}".format(x)])
            
        elif 'parameter' in out_types[x]: # non-linear parameters output
            branch_dic["branch{0}".format(x)] = LSTM(8, return_sequences=True)(pred)
            branch_dic["branch{0}".format(x)] = AveragePooling1D(ds_step)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = LSTM(4, return_sequences=True)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = AveragePooling1D(ds_step)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = LSTM(2, return_sequences=True)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = AveragePooling1D(ds_step)(branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = LSTM(1, return_sequences=True)(branch_dic["branch{0}".format(x)])
            
            amount_cat = extract_number(out_types[x]) # extracts number of categories from type-description
            print("Anzahl an parameter: ", amount_cat)
            
            branch_dic["branch{0}".format(x)] = Flatten()(branch_dic["branch{0}".format(x)])
            # Automate naming layer with out_types to ensure uniqueness
            branch_dic["branch{0}".format(x)] = Dense(amount_cat, activation='linear' , name='parameter_output')(branch_dic["branch{0}".format(x)])

    # Concating outputs
    if len(out_types)>1: # check if multiple feature in output of NN
        # concatenate layer of the branch outputs
        print("Branch Werte")
        print("List of branches", branch_dic.values())
        model = Model(Input_encoder, branch_dic.values())
    else: # single feature in output
        print("Branch Werte")
        print("List of branches", branch_dic.values())
        print(branch_dic["branch0"])
        model = Model(Input_encoder, branch_dic["branch0"])
        
    # Add loss manually, because we use a custom loss with global variable use
    # model.add_loss(lambda: my_loss_fn(y_true, con, OUTPUT_name))
    return model, ds_samplerate
    
def ECG_loss(y_true, y_pred):
        """
        Custom LOSS function

        Goal:  weighted sum of LOSSes of each Feature
                - Weights are for different columns of Output Tensor
                    - Weights decrease for each feature
                    - this way the gradient for each additional feature is more gentle compared to previous features
                    - this gives a loose order in which the LOSSes are minimized
        """
        squarred_differences = []
        mse = K.losses.MeanSquaredError() # function for LOSS of choice
        squarred_differences.append(tf.cast(mse(y_true[:,:], y_pred[:,:]),tf.float32)) # best results with multiplication of 10 to power of k with features from easy to difficult
        sd = tf.stack(squarred_differences) # Concat the LOSSes of each feature
        
        # Here we calculate the mean of each column
        loss = tf.reduce_mean(sd, axis=-1)  # Note the `axis=-1`
        
        return loss
    
def symbols_loss(y_true, y_pred):
    """Custom LOSS function für symbol classification
    
    Here we use Sparse Crossentropy for every time step seperately and sum them up
    For higher efficiency
    The function returns a mean of the losses
    """
    sce = K.losses.SparseCategoricalCrossentropy(from_logits=False) # function for LOSS of choice
    loss = tf.constant(0, dtype=tf.float16)
    ds_samplerate = int(2**7)
    copies = tf.constant(2, dtype=tf.int16) # number of copies added to examples per rare class
    additions = tf.constant(0, dtype=tf.int32) # total number of copies of rare classes added to examples
    # We calculate CrossEntropy of at every half second. This way we speed up training. There is a lot of redundancy in one second
    for time in tf.range(y_true.shape[1], delta=int(ds_samplerate/2)): # loop over time series with ds_samplerate steps
        # "2" and "0" occur rarely and need to be weighted heavier to be learned properly
        # For this we find the classes and construct a boolean mask
        zeros = tf.equal(y_true[:,time], tf.zeros(tf.shape(y_true[:,time]), dtype=tf.int32)) # find zero elements
        twos = tf.equal(y_true[:,time], tf.fill(tf.shape(y_true[:,time]), 2)) # find two elements
        rare_classes = tf.logical_or(zeros, twos) # combine both masks
        additions = tf.reduce_sum(tf.cast(rare_classes, tf.int32)) # we count how often we find "2" and "0"
        add_class_true = tf.repeat(tf.boolean_mask(y_true[:,time], rare_classes), copies) # create multiple copies of rare classes
        stack_true = tf.concat([y_true[:,time], add_class_true], axis=0) # stack copies onto original point in time
        add_class_pred = tf.repeat(tf.boolean_mask(y_pred[:,time], rare_classes), copies, axis=0) # create multiple copies of rare classes
        stack_pred = tf.concat([y_pred[:,time], add_class_pred], axis=0) # stack copies onto original point in time
        loss += sce(stack_true, stack_pred) # sparse crossentropy of one time step. Very important. Crossentropy not defined for whole time series
    
    # we normalize loss by size of batch and number of added copies
    # Additional we factor we loss with 500. This way it is comparable to the Words-distribution-loss
    return 1*(loss/tf.cast(y_true.shape[1] + additions/y_true.shape[1], tf.float16))
    
def extract_number(string):
    """ Extracts numeric elements from string as integer
    """
    # print(string)
    numeric_filter = filter(str.isdigit, string)
    numeric_string = "".join(numeric_filter)
    # print(numeric_string)
    return int(numeric_string)

def extract_letter(string):
    """ Extracts letter elements from string as integer
    """
    # print(string)
    letter_filter = filter(str.isalpha, string)
    letter_string = "".join(letter_filter)
    # print(numeric_string)
    return letter_string

def save_XY(X,y):
    """Saves X and y rowwise in npy files
    is needed in generator to save Cache space in GPU"""
    for ID in range(len(X[:,0])): # loop over all examples
        np.save('data/generator/X-'+ str(ID) + ".npy", X[ID])
        np.save('data/generator/y_ecg-'+ str(ID) + ".npy", y[0][ID])
        np.save('data/generator/y_bbi-'+ str(ID) + ".npy", y[1][ID])
    
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
        result = engine.calc_symboldynamics(BBI[row], 0.01, "movdiff", nargout=2)
        flatten = [result[1][n][0] for n in range(len(result[1]))] # flatten the list of matlab.doubles
        # print(np.array(flatten,dtype=int))
        symbols.append(np.array(flatten,dtype=int))
        words.append(result[0])
        
        # Here we will add non-linear parameters calculated from symbols and words
        
    words = np.array(words, dtype=int)[:,:,0] # transform list of distributions into numpy array for faster processing
    engine.quit()
    
    # Plot symbols and words
    example = np.random.randint(len(symbols[:]))
    
    plt.figure(1)
    plt.plot(list(range(len(symbols[example][:]))), symbols[example][:])
    plt.title("timeseries of symbols")
    plt.savefig("symbols.png")
    plt.close()
    
    plt.figure(2)
    plt.title("Distribution of words")
    plt.bar(np.linspace(0,64,num=64), words[example,:])
    plt.xlabel("time in ms")
    plt.ylabel("occurence")
    plt.savefig("words.png")
    plt.close()
    return symbols, words

def plvar(BBI):
    """Function to calculate plvar parameters of BBI
    Input: BBI. list of numpy.arrays with different length. timeseries of beat-to-beat-intervalls for each beat no interpolation
    
    returns: plvar_5_param, plvar_10_param, plvar_20_param. numpy array of np.float16"""
    future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
    engine = future.result() # run engine in background
    engine.cd(r'nl', nargout=0) # change directory to nl folder
    plvar_5_param = [] # list of parameters
    plvar_10_param  = [] # list of parameters
    plvar_20_param = [] # list of parameters
    for row in range(len(BBI[:])): # loop over examples
        # insert one-dim array from list into matlab function
        # function returns us parameter as output
        plvar_5_param.append(engine.plvar(BBI[row], 5, nargout=1))
        plvar_10_param.append(engine.plvar(BBI[row], 10, nargout=1))
        plvar_20_param.append(engine.plvar(BBI[row], 20, nargout=1))
    plvar_5_param = np.array(plvar_5_param, dtype=np.float16)
    plvar_10_param = np.array(plvar_10_param, dtype=np.float16)
    plvar_20_param = np.array(plvar_20_param, dtype=np.float16)
    
    return plvar_5_param, plvar_10_param, plvar_20_param

def phvar(BBI):
    """Function to calculate phvar parameters of BBI
    Input: BBI. list of numpy.arrays with different length. timeseries of beat-to-beat-intervalls for each beat no interpolation
    
    returns: phvar_20_param, phvar_50_param, phvar_100_param. numpy array of np.float16"""
    future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
    engine = future.result() # run engine in background
    engine.cd(r'nl', nargout=0) # change directory to nl folder
    phvar_20_param = [] # list of parameters
    phvar_50_param  = [] # list of parameters
    phvar_100_param = [] # list of parameters
    for row in range(len(BBI[:])): # loop over examples
        # insert one-dim array from list into matlab function
        # function returns us parameter as output
        phvar_20_param.append(engine.plvar(BBI[row], 5, nargout=1))
        phvar_50_param.append(engine.plvar(BBI[row], 10, nargout=1))
        phvar_100_param.append(engine.plvar(BBI[row], 20, nargout=1))
    phvar_20_param = np.array(phvar_20_param, dtype=np.float16)
    phvar_50_param = np.array(phvar_50_param, dtype=np.float16)
    phvar_100_param = np.array(phvar_100_param, dtype=np.float16)
    
    return phvar_20_param, phvar_50_param, phvar_100_param

def wsdvar(symbols):
    """Function to calculate wsdvar parameter of symbol dynamics
    Input: symbols. list of numpy.arrays with different length. timeseries of symbols for each beat
    
    returns: wsdvar. numpy array of np.float16"""
    future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
    engine = future.result() # run engine in background
    engine.cd(r'nl', nargout=0) # change directory to nl folder
    wsdvar_param = [] # list of parameters
    for row in range(len(symbols[:])): # loop over examples
        # insert one-dim array from list into matlab function
        # function returns us parameter as output
        wsdvar_param.append(engine.wsdvar(symbols[row], nargout=1))
    wsdvar_param = np.array(wsdvar_param, dtype=np.float16)
    
    return wsdvar_param

def words_parameters(words):
    """Function to calculate 
    - forbidden words parameter 
    - shannon entropy
    - renyi entropy
    - wpsum02
    - wpsum13
    
    of words distribution
    Input: words. numpy.array. distributions of 3 letter words of 4 category words
    
    returns:    forbword. numpy array of np.float16
                fwshannon_param. numpy array of np.float16
                fwrenyi_025_param, fwrenyi_4_param. numpy array of np.float16
                wpsum02_param. numpy array of np.float16
                wpsum13_param. numpy array of np.float16"""
    future = matlab.engine.start_matlab(background=True) # asynchrounus run of matlab engine
    engine = future.result() # run engine in background
    engine.cd(r'nl', nargout=0) # change directory to nl folder
    forbword = [] # list of parameters
    fwshannon_param = [] # list of parameters
    fwrenyi_025_param = [] # list of parameters
    fwrenyi_4_param = [] # list of parameters
    wpsum02_param = [] # list of parameters
    wpsum13_param = [] # list of parameters
    for row in range(len(words[:])): # loop over examples
        # insert one-dim array from list into matlab function
        # function returns us parameter as output
        forbword.append(engine.forbidden_words(words[row], nargout=1))
        fwshannon_param.append(engine.fwshannon(words[row].astype(np.float64), nargout=1)) # Very important matlab functions defined for float64 / double
        fwrenyi_025_param.append(engine.fwrenyi(words[row].astype(np.float64), 0.25, nargout=1))
        fwrenyi_4_param.append(engine.fwrenyi(words[row].astype(np.float64), 4., nargout=1))
        wpsum02_param.append(engine.wpsum02(words[row], nargout=1))
        wpsum13_param.append(engine.wpsum13(words[row], nargout=1))
    wpsum13_param = np.array(wpsum13_param, dtype=np.float16)
    wpsum02_param = np.array(wpsum02_param, dtype=np.float16)
    fwrenyi_025_param = np.array(fwrenyi_025_param, dtype=np.float16)
    fwrenyi_4_param = np.array(fwrenyi_4_param, dtype=np.float16)
    fwshannon_param = np.array(fwshannon_param, dtype=np.float16)
    forbword = np.array(forbword, dtype=np.float16)
    
    return forbword, fwshannon_param, fwrenyi_025_param, fwrenyi_4_param, wpsum02_param, wpsum13_param

def cut_BBI(data, lag, length_item):
    """Function to cut interval of interest from BBI timeseries and return symbols and words
    """
    # wir nutzen Funktionen geschrieben in Matlab
    # Gecodet von matthias
    
    # Cutting BBI fitting our needs
    # lower bound / starting point of BBI time series. Defined by lag in samples / data points
    lb_BBI = np.where(np.cumsum(data, axis=1) >= float(lag[4:])/samplerate*1000, data, 0)
    # upper bound / ending point of BBI
    up_BBI = np.where(np.cumsum(lb_BBI, axis=1) <= length_item/samplerate*1000, lb_BBI, 0)
    BBI = up_BBI[:, ~np.all(up_BBI == 0, axis = 0)] # cut all columns with only zeros out
    BBI_list = [] # saving in a list allows different length of sequences
    for n in range(len(BBI[:,0])):
        bbi = BBI[n,:]
        BBI_list.append(bbi[~(bbi==0)])
    
    # Extracting symbols from cut BBI
    symbols, words = calc_symboldynamics(BBI_list) # Outputs lists of arrays. Symbols have different length
    return symbols, words, BBI_list