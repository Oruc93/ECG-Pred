# This library if for processing an ECG to BBI, words, symbols and non-linear parameters

import os
from random import sample
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from keras.layers import Dense, Input, BatchNormalization,  LSTM, TimeDistributed
from keras.layers import Conv1D, Conv1DTranspose, UpSampling1D, AveragePooling1D, MaxPooling1D, Reshape, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
import tensorflow as tf
import tensorflow.keras as K

# print("Tensorflow version: ", tf.__version__)

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
    """Memorizes needed data from HDF5 file
    Returns as numpy array
    
    HDF5 is named in filename
    dataset is a list of timeseries and parameters we are interested in
    
    returns data. Dictionary with timeseries and parameters as keys and datapoints as values
    values can be sequences or singular values
    """
    # Checks if filename is a string
    if not(isinstance(filename, str)):
        print("filename must be a string naming a H5-file.")
    # Checks if dataset is a string
    if not(isinstance(dataset, list)):
        print("dataset must be a list of keys as datasets of the H5-file.")
    
    try:
        file = h5py.File(filename, "r") # pointer to h5-file
    except:
        print("H5-file not found in current directory. Choose one of the files below")
        print("Current directory:", dname)
        print([".data/" + name for name in os.listdir("./data") if "h5" in name]) # listet alle h5 Dateien im data Ordner
        print([".data/" + name for name in os.listdir("./data") if "hdf5" in name])
        print([".data/" + name for name in os.listdir() if "h5" in name]) # listet alle h5 Dateien im aktuellen Ordner
        print([".data/" + name for name in os.listdir() if "hdf5" in name])
    
    data = {} # dictionary containing data
    for name in dataset: # loop over datasets named in list
        print("Memorizing timeseries: ", name)
        try:
            data[name] = file[name][:] # loads pointer to dataset into variable
            print("Shape of " + name, np.shape(data[name]))
        except:
            if name=='Tachy': # we will use Tachygramm with x-axis in samples
                data[name] = file['RP'][:] # loads pointer to dataset into variable
                print("Shape of " + name, np.shape(data[name]))
            else:
                print("dataset not found in H5-file. Choose one of the datasets below")
                print(file.keys())
    # print("Available keys: ", file.keys())
    
    # plots the distribution of first dataset
    plt.figure(1)
    plt.title("Distribution of data " + dataset[0])
    data_flat = data[dataset[0]].flatten()
    plt.hist(data_flat, bins=30)
    plt.xlabel("time in ms")
    plt.ylabel("occurence")
    plt.savefig("distribution of data")
    plt.close()
    
    global samplerate
    samplerate = file.attrs['samplerate'][0]
    
    return data, samplerate # returns datasets

def set_items(data, INPUT_name, OUTPUT_name, length_item):
    """ This function pulls items from datasets
     The items are separated by Input and Output, training and test
     
     
    :param data: dictionary containing datasets
    :param INPUT_name: selection of input features and lags. Dictionary
    :param OUTPUT_name: selection of output features and lags. Dictionary
    :param length_item: length of items given into the neural network
    
    :return X,y: numpy arrays of training or test sets
     """    
    # Constructing toy problem sinus curve in shape of data
    """x = np.linspace(0, np.shape(data)[1]/250, num=np.shape(data)[1])
    for k in range(np.shape(data)[0]):
        ra = np.random.rand(1)*np.pi
        data[k, :] = np.sin(x + ra)"""
        
    # Min-Max Scaling time series in data
    """for k in range(len(data[:,0])): # loop over all time series
        data[k,:] = (data[k,:] - np.min(data[k,:])) / (np.max(data[k,:]) - np.min(data[k,:])) # min-max scaling"""
        
    # Calculation of the features and slicing of time series
    print("Constructing input array ...")
    X = constr_feat(data, INPUT_name, length_item)
    X = X[list(X.keys())[0]] # Takes value out of only key of dictionary
    print(np.shape(X))
    print("Constructing output array ...")
    out_types = output_type(data, OUTPUT_name) # global list of types of output for loss and output layer
    y = constr_feat(data, OUTPUT_name, length_item)
    y_list = [] # Sorting values of dictionary into list. Outputs need to be given to NN in list. Allows different formatted Truths
    for key in y.keys():
        y_list.append(y[key])
    return X, y_list, out_types

def output_type(data, OUTPUT_name):
    """ returns list with type of output at the corresponding column
    Types are of regressional and categorical timeseries and parameters
    The three types need to be handled differently in the output layer and loss function to get the desired results
    
    returns: list. Types of output for each column
    """
    dic_types = {
        "ECG": "regressionECG", "MA": "regressionMA", "RP": "classificationRP", "BBI": "regressionBBI", "symbols": "classificationS", "words": "distributionW",
        "Tachy": "regressionTachy",
        "forbword": "parameter", "fwshannon": "parameter", "fwrenyi 0.25": "parameter",
        "fwrenyi 4": "parameter", "wsdvar": "parameter", "wpsum 02": "parameter", 
        "wpsum 13": "parameter", "plvar 5": "parameter", "plvar 10": "parameter", 
        "plvar 20": "parameter", "phvar 20": "parameter", "phvar 50": "parameter", "phvar 100": "parameter"
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
                if key == 'symbols': # Check ob Symbole einbezogen werden
                    for k in range(len(np.unique(data[key][:]))): # unique labels in timeseries
                        out_types.append(dic_types[key] + str(k)) # number columns containing label
                        # Important for initializing model, loss function and plotting output
                elif key == 'words':
                    for k in range(len(data[key][0])): # unique labels in timeseries
                        out_types.append(dic_types[key] + str(k)) # number columns containing label
                        # Important for initializing model, loss function and plotting output
                else:
                    out_types.append(dic_types[key]) # regression and parameter timeseries
            except:
                out_types.append("No type detected")
    return out_types

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def constr_feat(data, NAME, length_item):
    """ This function constructs the feature from given datasets and lags of interest
    If given a timeseries it transforms it to a sequence with lag
    If given parameter it transform into sequence with constant value
    
    :param data: dictionary containing dataset
    :param NAME: list of features. list of strings
    :param length_item: length of items given into the neural network
    :return sequence: numpy arrays of features. rows timeseries. Columns features
     """
    # print("Data: ", data)
    print(NAME)
    count = 0
    for key in NAME: # loop über Zeitreihen von Interesse
        if isinstance(NAME[key], list): # Zählen der Anzahl an lag-Versionen der Zeitreihen
            amount_cat = 1 # reset of category variable
            if key == 'symbols': # Check ob Symbole einbezogen werden
                amount_cat = len(np.unique(data[key][:])) # Anzahl an Kategorien in Zeitreihe
            if key == 'words': # Check ob Symbole einbezogen werden
                amount_cat = len(data[key][0]) # Anzahl an Kategorien in Zeitreihe
            count += len(NAME[key]) * amount_cat # Summation der Menge an einbezogenen Zeitreihen
        else:
            Warning("Value of key is not a list", key)
    print("Number of columns: ", count)
    # print(data)
    sequence = np.zeros((np.shape(data[key])[0], length_item, count)) # prebuilding empty sequence
    dic_seq = {}
    global BBI_size # saves size of BBI arrays globally. Is needed in NN construction
    if not('BBI_size' in globals()): # check if BBI_size exists globally
        BBI_size = 0
    feat_number = int(-1)
    for key in NAME: # loop over dictionary keys
        for name in NAME[key]: # loop over lags and options found under key
            feat_number += 1
            # check if data is timeseries or parameter
            print("Shape of data ", key, " : ", np.shape(data[key]))
            if len(data[key][0,:]) > 1: # timeseries is saved with lag
                print("timeseries ", key," with lag ", int(name[4:]), " at column ", feat_number)
                
                # BBI time series ändern
                # x-Achse in samples. Problem der Arraygröße verschwindet
                # bei downgesamplter Größe bleiben im Netz
                # zwischen den BBI linear interpolieren
                if key == 'BBI':
                    # Main problem is the different length BBI and ecg timeseries
                    # First try with padded sequence. This way output data has fixed size
                    # We pad before the time series
                    # slice correct BBI of interval of interest
                    # data[key] *= 0.001 # scale values form ms to s. network learns better this way
                    print("feature: lag ", int(name[4:]), "at column ", feat_number) # lag is in given in samples / datapoints
                    # lower bound / starting point of BBI time series. Defined by lag in samples / data points
                    # lb_BBI = np.where(np.cumsum(data[key], axis=1) >= float(name[4:])/samplerate*1000, data[key], 0)
                    # # upper bound / ending point of BBI
                    # up_BBI = np.where(np.cumsum(lb_BBI, axis=1) <= length_item/samplerate*1000, lb_BBI, 0)
                    # BBI = up_BBI[:, ~np.all(up_BBI == 0, axis = 0)] # cut all columns with only zeros out
                    # if BBI_size == 0: # check if BBI_size was set
                    #     pad_size = 1 # BBI_size was never sat. good padding to include worst case
                    # else:
                    #     pad_size = BBI_size-len(BBI[0,:]) # padding Test data to training size
                    #     if pad_size<0: # check if worst case was pessimist enough
                    #         SyntaxError("pad_size four lines above is too small. Raise by: ", -pad_size)
                    # BBI = np.concatenate((np.zeros((np.shape(BBI)[0], pad_size)), BBI), axis=1) # pad zeros to front of time series
                    # for row in range(len(BBI[:,0])): # loop over all rows in BBI array
                    #     while True: # loop until all time series are flushed to right
                    #         if BBI[row,-1]==0: # check if last value of BBI is zero
                    #             BBI[row,:] = np.roll(BBI[row,:],1) # moves all element 1 to the right. Right most moves to first
                    #         else:
                    #             break # last element non-zero -> row is flushed to right -> break while loop
                    # dic_seq[key+name] = BBI * 0.001 # scale values form ms to s. network learns better this way
                    # print(np.shape(dic_seq[key+name]))
                    # BBI_size = max(BBI_size, len(dic_seq[key+name][0,:]))
                
                if key == 'Tachy':
                    # Tachygram of ecg. x-axis: sample. y-axis: ms
                    print("feature: lag ", int(name[4:]), "at column ", feat_number) # lag is in given in samples / datapoints
                    bc = data[key][:, int(name[4:]) : length_item + int(name[4:])]
                    bc = bc==3 # binary categorizing of r-peaks
                    rp = np.argwhere(bc>0) # position of r-peaks in samples of all examples
                    ds_samplerate = int(2**7) # Ziel samplerate beim Downsampling
                    ratio = samplerate / ds_samplerate # quotient between both samplerates
                    tachy = np.zeros((len(bc[:,0]), int(length_item / ratio))) # empty array to contain tachygram
                    for n in range(len(bc[:,0])): # loop over all examples
                        ts =  np.argwhere(rp[:,0]==n)[:,0] # position of r-peaks of example n in rp
                        rp_ts = rp[ts,1] # position of r-peaks in example n
                        y_ts = rp_ts[1:] - rp_ts[:-1] # calculate BBI between r-peaks. Exlude first point
                        tachy[n,:] = np.interp(list(range(len(tachy[0,:]))), rp_ts[1:] / ratio, y_ts) # position r-peaks and interpolate tachygram. x-axis in samples
                    tachy = tachy / samplerate # transform from sample into ms
                    dic_seq[key+name] = tachy
                    plt.figure(1)
                    # plt.plot(list(range(len(tachy[n,:]))), tachy[n,:])
                    plt.plot(np.linspace(0, len(tachy[n,:]) / ds_samplerate, num=len(tachy[n,:])), tachy[n,:])
                    plt.savefig("Tachygram.png")
                    plt.close()
                    
                if key == 'RP': # position of r-Peak
                    
                    # Funktioniert nicht !!!!
                    
                    # time series in ipeaks contains categories of waves and peaks of ecg
                    # each sample has a categorical value
                    # ipeaks: labels for PQRST peaks: P(1), Q(2), R(3), S(4), T(5)
                    # A zero lablel is output otherwise ... to fin R-peaks use sequence == 3
                    print("feature: lag ", int(name[4:]), "at column ", feat_number) # lag is in given in samples / datapoints
                    bc = data[key][:, int(name[4:]) : length_item + int(name[4:])]
                    bc = bc==3 # binary categorizing of r-peaks
                    ds_samplerate = int(2**7) # Ziel samplerate beim Downsampling
                    ds = int(samplerate/ds_samplerate) # downsampling ratio
                    # seq = np.full((len(bc[:,0]), int(len(bc[0,:])/ds)), False) # True False classification
                    seq = np.full((len(bc[:,0]), int(len(bc[0,:])/ds)), 0.) # 0 1 classification float
                    gauss = gaussian(np.linspace(-3, 3, ds*5), 0, 1) * 1. # gaussian to overlay over r-peak
                    for example in range(len(bc[:,0])): # loop over all examples
                        for sample in range(len(seq[0,:])): # loop over all downsamples
                            # seq[example, sample] = np.any(bc[example, sample*ds:sample*ds+ds-1]) # window of 8 samples. If any is true, seq is true
                            if np.any(bc[example, sample*ds:sample*ds+ds-1]): # window of 8 samples. If any is 1, seq is 1
                                seq[example, sample] = 1.
                        conv_seq = np.convolve(seq[example, :], gauss, mode='same')
                        # print(np.shape(conv_seq))
                        seq[example, :] += conv_seq
                        seq[example, :] = np.clip(seq[example, :], 0, 1)
                    dic_seq[key+name] = seq
                    print(np.shape(dic_seq[key+name]))
                if key == 'words': # transform categories of words into timeseries length
                    
                    # diese categorisierung nachbesser
                    
                    amount_cat = len(data[key][0]) # Anzahl an Kategorien in Zeitreihe
                    # print(data[key][0])
                    # print(np.shape(data[key][0]))
                    for n in range(len(sequence[:,0,0])): # loop over all examples
                        sequence[n,:,feat_number:feat_number+amount_cat] = np.full(
                            np.shape(sequence[n,:,feat_number:feat_number+amount_cat]),
                            data[key][n]
                        )
                    feat_number += amount_cat - 1
                if key == 'symbols': # Transform into one-hot vector
                    
                    # diese categorisierung nachbesser
                    # ist noch von BBI analyse
                    # sprich jede x-Stelle ist einem Beat zugeordnet
                    
                    sequence[:,:,feat_number] = data[key][:, int(name[4:]) : length_item+int(name[4:])]
                    amount_cat = len(np.unique(data[key][:])) # Anzahl an Kategorien in Zeitreihe
                    seq_sy = np.zeros((np.shape(data[key])[0], length_item, amount_cat)) # prebuilding empty sequence
                    layer = K.layers.CategoryEncoding(num_tokens = amount_cat, output_mode="one_hot")
                    for n in range(len(seq_sy[:,0,0])):
                        seq_sy[n,:,:] = layer(sequence[n,:,feat_number])
                    sequence[:,:, feat_number:feat_number+amount_cat] = seq_sy
                    feat_number += amount_cat - 1
                if key == 'ECG': # check if feature is of lag nature
                    print("feature: lag ", int(name[4:]), "at column ", feat_number)
                    # sequence[:,:,feat_number] = data[key][:, int(name[4:]) : length_item + int(name[4:])]
                    # sequence[:,:,feat_number] *= 100 # min-max scaling by hand
                    seq = data[key][:, int(name[4:]) : length_item + int(name[4:])]
                    seq *= 100 # min-max scaling by hand
                    dic_seq[key+name] = seq
                    print(np.shape(dic_seq[key+name]))                    
                    continue
                # calculation with convolution of padded interval and sequence of ones
                if key == 'moving average': # Calculating moving average of half second before and after. Here: 1000Hz
                    # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
                    print("feature: moving average at column ", feat_number)
                    ma_win = float(name) # window of MA in seconds
                    s = int(samplerate*ma_win)
                    h_s = int(s/2) # half of the window length. important to for getting padding around interval of interest
                    # padding with points from timeseries. This way more information in feature
                    for n in range(np.shape(sequence)[0]): # loop over ecgs
                        # convolution of interval of interest and sequence of ones with length samplerate
                        # need padding for first h_s values of MA. First values are not vital for our analysis
                        window = np.concatenate((np.zeros(h_s), data[key][n, 0 : length_item + h_s]))
                        ma = np.convolve(window, np.ones(s), 'valid') / s # convolute moving average
                        sequence[n, :, feat_number] = ma[1:] # first value is discarded
                    continue
            else: # parameter is repeated to length of item
                print("parameter ", key," with lag ", int(name[4:]), " at column ", feat_number)
                sequence[:,:,feat_number] = np.repeat(data[key][:], length_item, axis=0)
                continue
    if key=='ECG' or key=='BBI' or key=='RP' or key=='Tachy':
        return dic_seq
    else:
        return sequence

def kernel_check(X):
    """Plots a 2s snippet of ecg
    We convolute 2s snippets out of whole ecg data in NN"""
    plt.figure(1)
    # print("Shape of first Convolution Kernel", np.shape(X[0,-2*samplerate:]))
    plt.plot(list(range(len(X[0,-2*samplerate:]))), X[0,-2*samplerate:])
    plt.title("Convolution with 2s-Kernels so they will look like this")
    plt.savefig("Conv-Kernels.png")
    plt.close()

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
        # print(np.shape(y[n]))
        data = y[n]
        plt.plot(list(range(len(data[0,:]))), data[0,:] + k)
        k = k + 0.0005
    plt.savefig("Test-y" + str(n) +".png")
    plt.close()

def check_data(X,y,X_test,y_test):
    # Check of Dataset
    plt.figure(1)
    plt.title("Examples from datasets 1000Hz")
    plt.plot(list(range(len(X[0,:]))), X[0,:])
    list_keys = list(y.keys())
    print(list(y.keys()))
    plt.plot(list(range(len(y[list_keys[0]][0,:]))), y[list_keys[0]][0,:])
    # plt.plot(list(range(len(X_test[0,:,0]))), X_test[0,:,0])
    # plt.plot(list(range(len(y_test[0,:,0]))), y_test[0,:,0])
    plt.legend(["X", "y", "X_test", "y_test"])
    plt.savefig("Full-Plot Xy datasets")
    plt.close()
    
    # print("Mittelwert")
    MEAN = np.mean(X,axis=0)
    VAR = np.var(X,axis=0)
    plt.figure(1)
    plt.title("Examples from datasets")
    # plt.plot(list(range(len(MEAN))), MEAN)
    plt.plot(list(range(len(VAR))), VAR)
    # plt.plot(list(range(len(y[0,:,0]))), y[0,:,0])
    # plt.plot(list(range(len(X_test[0,:,0]))), X_test[0,:,0])
    # plt.plot(list(range(len(y_test[0,:,0]))), y_test[0,:,0])
    plt.legend(["MEAN", "VAR", "X_test", "y_test"])
    plt.savefig("Mittelwert")
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
    ds_step = 2 # factor of down- and upsampling of ecg timeseries
    ds_samplerate = int(2**7) # Ziel samplerate beim Downsampling
    orig_a_f = int(2**4) # first filter amount. low amount of filters ensures faster learning and training
    amount_filter = orig_a_f
    # encoder = AveragePooling1D(ds_step)(Input_encoder)
    encoder = Conv1D(amount_filter, # number of columns in output. filters
                     samplerate*2, # kernel size. We look at 2s snippets
                     # strides = 1, # we convolute with overlapping kernel. This way correlation is kept for sure
                     padding = "same", # only applies kernel if it fits on input. No Padding
                     # kernel_initializer = my_init, # custom kernel initializer
                     dilation_rate=2, # Kernel mit regelmäßigen Lücken. Bsp. jeder zweite Punkt wird genommen
                     activation = "relu"
                     )(Input_encoder)  # downgrade numpy to v1.19.2 if Tensor / NumpyArray error here
    encoder = AveragePooling1D(ds_step)(encoder)
    
    # our hidden layer / encoder
    # decreasing triangle
    orig_size = size  # size of layer in branches before output / decoder architecture
    k = ds_step
    print("Downsampled to: ", int(samplerate/k), " Hz") # A samplerate under 100 Hz will decrease analysis quality. 10.4258/hir.2018.24.3.198
    while ds_samplerate < int(samplerate/k):
        # size = int(size / 2)
        amount_filter /= 2
        # encoder = AveragePooling1D(ds_step)(encoder)
        encoder = Conv1D(amount_filter, # number of columns in output. filters
                     int(samplerate/k), # kernel size. We look at 2s snippets
                     # strides = 2, # we convolute with overlapping kernel. This way correlation is kept for sure
                     padding = "same", # only applies kernel if it fits on input. No Padding
                     # kernel_initializer = my_init, # custom kernel initializer
                     dilation_rate=2,
                     activation = "relu"
                     )(encoder)  # downgrade numpy to v1.19.2 if Tensor / NumpyArray error here
        encoder = AveragePooling1D(ds_step)(encoder)
        
        k *= ds_step
        print("Downsampled to: ", int(samplerate/k), " Hz")
    
    encoder = Dense(size)(encoder)
    # LSTM branch
    lstm_br = LSTM(size, return_sequences=True)(encoder) # vielleicht amount_filter statt size
    # Dense branch
    dense_br = Dense(size)(encoder)
    # concat
    con_br = concatenate([lstm_br, dense_br])
    pred = Dense(size)(con_br)
    
        
    # branching of the pseudo-tasks
    # expanding triangle / decoder until all branches combined are as wide as the input layer
    branch_dic = {}  # dictionary for the branches
    latent_a_f = amount_filter
    for x in range(len(out_types)):
        amount_filter = latent_a_f
        if 'ECG' in out_types[x]:        
            branch_dic["branch{0}".format(x)] = Conv1D(amount_filter,
                                                        1,
                                                        strides=1,
                                                        padding = "same",
                                                        activation="relu")(
                                                        pred)
            branch_dic["branch{0}".format(x)] = UpSampling1D(ds_step)(branch_dic["branch{0}".format(x)])
            amount_filter *= 2
            while amount_filter <= orig_a_f:
                branch_dic["branch{0}".format(x)] = Conv1D(amount_filter,
                                                            1,
                                                            strides=1,
                                                            padding = "same",
                                                            activation="relu")(
                                                            branch_dic["branch{0}".format(x)])
                branch_dic["branch{0}".format(x)] = UpSampling1D(ds_step)(branch_dic["branch{0}".format(x)])
                amount_filter *= 2
            branch_dic["branch{0}".format(x)] = Conv1D(1,
                                                        1,
                                                        strides=1,
                                                        padding = "same",
                                                        name = "ECG_output",
                                                        activation="linear")( # sigmoid for between 0 and 1
                                                        branch_dic["branch{0}".format(x)])
        elif 'BBI' in out_types[x]: # Das hier neumachen, nachdem constr_feat gemacht wurde
            branch_dic["branch{0}".format(x)] = LSTM(size)(pred)
            branch_dic["branch{0}".format(x)] = Dense(BBI_size, activation="linear", name="BBI_output")(branch_dic["branch{0}".format(x)])
        elif 'RP' in out_types[x]: # position of R-peak output
            branch_dic["branch{0}".format(x)] = LSTM(size, return_sequences=True)(pred)
            branch_dic["branch{0}".format(x)] = Dense(1, activation="sigmoid", name="RP_output")(branch_dic["branch{0}".format(x)])
        elif 'Tachy' in out_types[x]: # Tachygram regression output
            branch_dic["branch{0}".format(x)] = LSTM(size, return_sequences=True)(pred)
            branch_dic["branch{0}".format(x)] = Dense(1, activation="linear", name="Tachy_output")(branch_dic["branch{0}".format(x)])

    
    # Concating outputs
    if len(out_types)>1: # check if multiple feature in output of NN
        # concatenate layer of the branch outputs
        print("Branch Werte")
        print(branch_dic.values())
        # con = concatenate(branch_dic.values())
        # dense calculates, so we use it in the branches
        # this way each branch is independent from each other in the decoding part
        # and focuses on its given feature
        model = Model(Input_encoder, branch_dic.values())
    else: # single feature in output
        print("Branch Werte")
        print(branch_dic.values())
        print(branch_dic["branch0"])
        model = Model(Input_encoder, branch_dic["branch0"])
        
    # Add loss manually, because we use a custom loss with global variable use
    # model.add_loss(lambda: my_loss_fn(y_true, con, OUTPUT_name))
    return model, ds_samplerate

def setup_maxKomp_Conv_AE_LSTM_P(input_shape, number_feat, samplerate, size=2**2):
    """ This NN has the goal to maximize the compression in the latent space
    
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
    ds_step = 2 # factor of down- and upsampling of ecg timeseries
    ds_samplerate = int(2**7) # Ziel samplerate beim Downsampling. 128Hz wird fürs erste als untere Grenze angenommen
    orig_a_f = int(2**4) # first filter amount. low amount of filters ensures faster learning and training
    amount_filter = orig_a_f
    encoder = Conv1D(amount_filter, # number of columns in output. filters
                     samplerate*2, # kernel size. We look at 2s snippets
                     # strides = 1, # we convolute with overlapping kernel. This way correlation is kept for sure
                     padding = "same", # only applies kernel if it fits on input. No Padding
                     # kernel_initializer = my_init, # custom kernel initializer
                     dilation_rate=2, # Kernel mit regelmäßigen Lücken. Bsp. jeder zweite Punkt wird genommen
                     activation = "relu"
                     )(Input_encoder)  # downgrade numpy to v1.19.2 if Tensor / NumpyArray error here
    encoder = AveragePooling1D(ds_step)(encoder)
    
    # our hidden layer / encoder
    # decreasing triangle
    orig_size = size  # size of layer in branches before output / decoder architecture
    k = ds_step
    print("Downsampled to: ", int(samplerate/k), " Hz") # A samplerate under 100 Hz will decrease analysis quality. 10.4258/hir.2018.24.3.198
    while ds_samplerate < int(samplerate/k):
        amount_filter /= 2 # lowest amount of filter with good result is 4
        encoder = Conv1D(amount_filter, # number of columns in output. filters
                     int(samplerate/k), # kernel size. We look at 2s snippets
                     padding = "same", # only applies kernel if it fits on input. No Padding
                     dilation_rate=2,
                     activation = "relu"
                     )(encoder)  # downgrade numpy to v1.19.2 if Tensor / NumpyArray error here
        encoder = AveragePooling1D(ds_step)(encoder)
        
        k *= ds_step
        print("Downsampled to: ", int(samplerate/k), " Hz")
    
    pred = Dense(size)(encoder)
    pred = LSTM(size, return_sequences=True)(pred) # vielleicht amount_filter statt size
    pred = Dense(size)(pred)
    # size so klein wie möglich halten und damit Anzahl Parameter am Flaschenhals hier verringern
    # NN zur Kompression zwingen und damit höheres Verständnis bei erfolgreicher Bearbeitung
    
        
    # branching of the pseudo-tasks
    # expanding triangle / decoder until all branches combined are as wide as the input layer
    branch_dic = {}  # dictionary for the branches
    latent_a_f = amount_filter
    for x in range(number_feat):
        amount_filter = latent_a_f
        branch_dic["branch{0}".format(x)] = Conv1D(amount_filter,
                                                    1,
                                                    strides=1,
                                                    padding = "same",
                                                    activation="relu")(
                                                    pred)
        branch_dic["branch{0}".format(x)] = UpSampling1D(ds_step)(branch_dic["branch{0}".format(x)])
        amount_filter *= 2
        while amount_filter <= orig_a_f:
            branch_dic["branch{0}".format(x)] = Conv1D(amount_filter,
                                                        1,
                                                        strides=1,
                                                        padding = "same",
                                                        activation="relu")(
                                                        branch_dic["branch{0}".format(x)])
            branch_dic["branch{0}".format(x)] = UpSampling1D(ds_step)(branch_dic["branch{0}".format(x)])
            amount_filter *= 2
        branch_dic["branch{0}".format(x)] = Conv1D(1,
                                                    1,
                                                    strides=1,
                                                    padding = "same",
                                                    activation="linear")( # sigmoid for between 0 and 1
                                                    branch_dic["branch{0}".format(x)])
    print("number of feat: ",number_feat)
    if number_feat>1: # check if multiple feature in output of NN
        # concatenate layer of the branch outputs
        print("Branch Werte")
        print(branch_dic.values())
        con = concatenate(branch_dic.values())
        # dense calculates, so we use it in the branches
        # this way each branch is independent from each other in the decoding part
        # and focuses on its given feature
        model = Model(Input_encoder, con)
    else: # single feature in output
        print("Branch Werte")
        print(branch_dic.values())
        print(branch_dic["branch0"])
        model = Model(Input_encoder, branch_dic["branch0"])
    return model, ds_samplerate

def my_loss_fn(y_true, y_pred):
        """
        Custom LOSS function

        Goal:  weighted sum of LOSSes of each Feature
                - Weights are for different columns of Output Tensor
                    - Weights decrease for each feature
                    - this way the gradient for each additional feature is more gentle compared to previous features
                    - this gives a loose order in which the LOSSes are minimized
        """
        squarred_differences = []
        # print("v value", v.eval())
        # Here we calculate the squared difference of each timestep of each sample in each column and weight it
        # for k in range(len(y_true[0,0,:])): # loop over columns in output array
        mse = K.losses.MeanSquaredError() # function for LOSS of choice
        # Calculate LOSS for each column and weight the result
        # append über tensorflow und unterschied in zeit messen
        squarred_differences.append(tf.cast(mse(y_true[:,:], y_pred[:,:]),tf.float32) * 10 ** 0) # best results with multiplication of 10 to power of k with features from easy to difficult
        # tf.print(v, output_stream=sys.stderr)
        # uarred_differences.write(k, mse(y_true[:,:,k], y_pred[:,:,k]))
        # tf.print("Loss individual")
        # tf.print(uarred_differences.read(k))
        # tf.print(squarred_differences)
        sd = tf.stack(squarred_differences) # Concat the LOSSes of each feature
        # tf.print("Loss stacked")
        # tf.print(uarred_differences.stack())
        # tf.print(sd)
        # print("sd dimension", K.backend.int_shape(sd))
        # Here we calculate the mean of each column
        loss = tf.reduce_mean(sd, axis=-1)  # Note the `axis=-1`
        # tf.print("Loss sum mean")
        # tf.print(loss)
        # tf.print(tf.reduce_mean(uarred_differences.stack(), axis=-1))
        # loss = tf.reduce_mean(squarred_differences.stack(), axis=-1)
        # print("Loss dimension", K.backend.int_shape(loss))
        return loss
    
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
    
def BBI_loss(y_true, y_pred):
        
        # Boolean maske entfernen, damit mit geänderter X-Achse arbeiten kann
        
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
        weights = tf.not_equal(y_true, tf.zeros(tf.shape(y_true), dtype=tf.float16)) # sets weights one for non-zero elements
        # weights = tf.cast(weights, tf.float16)
        # sd = tf.stack(squarred_differences*weights) # Concat the LOSSes of each feature
        sd = tf.stack(tf.boolean_mask(squarred_differences, weights))
        
        # Here we calculate the mean of each column
        loss = tf.reduce_mean(sd, axis=-1)  # Note the `axis=-1`
        
        return loss
    
def RP_loss(y_true, y_pred):
    
        # Label 1 stärker gewichten. Vielleicht nur diese Bewerten mit Bool Maske
    
        """
        Custom LOSS function

        Goal:  weighted sum of LOSSes of each Feature
                - Weights are for different columns of Output Tensor
                    - Weights decrease for each feature
                    - this way the gradient for each additional feature is more gentle compared to previous features
                    - this gives a loose order in which the LOSSes are minimized
        """
        
        bce = K.losses.BinaryCrossentropy(from_logits=True) # function for LOSS of choice
        # loss = bce(y_true, y_pred)
        # Apply the weights
        # weight_vector = tf.cast(y_true, tf.float16) * 0.999 + (1. - tf.cast(y_true, tf.float16)) * 0.001
        weight_vector = tf.math.rint(y_true) * 0.5 + (1. - tf.math.rint(y_true)) * 0.5
        # loss = weight_vector * b_ce
        
        mse = K.losses.MeanSquaredError() # function for LOSS of choice
        loss = weight_vector * mse(y_true, y_pred)
        
        # # sd = tf.stack(bce(y_true[:,:], y_pred[:,:])) # Concat the LOSSes of each feature
        # R_true = y_true[y_true==True]
        # # tf.print("Shape of R_true", tf.shape(R_true))
        # R_pred = y_pred[y_true==True]
        # loss = bce(R_true, R_pred)**2*10**3
        
        # R_true = y_true[y_true==False]
        # # tf.print("Shape of R_true", tf.shape(R_true))
        # R_pred = y_pred[y_true==False]
        # loss = bce(R_true, R_pred)
        
        # pR_true = y_true[y_pred>0.5]
        # # tf.print("Shape of R_true", tf.shape(R_true))
        # pR_pred = y_pred[y_pred>0.5]
        # loss += bce(pR_true, pR_pred)
        # # loss = bce(y_true[:,:], y_pred[:,:])
        
        # Here we calculate the mean of each column
        # loss = tf.reduce_mean(sd, axis=-1)  # Note the `axis=-1`
        
        return loss
    
def index(y):
    """transforms one-hot vector into index vector
    
    y: numpy array with size (sample, length_item, labels)
    
    returns:    y_i: numpy array with size(sample, length_item)
                hist: numpy array. distribution feature
    """
    k = int(-1)
    y_i = []
    while k < len(y[0,0,:])-1: # loop over columns
        k += 1
        if "classificationS" in out_types[k]:
            amount_cat = count_cat(k) # count amount of categories in current classification time series
            y_i.append(y[:,:,k:k+amount_cat].argmax(axis=2)) # one-hot vector to classification time series
            k += amount_cat -1
        elif "distributionW" in out_types[k]: # detects distribution feature
            amount_cat = count_cat(k) # count amount of categories in current classification time series
            hist =y[:,-1,k:k+amount_cat] # slicing last time step as distribution
            k += amount_cat -1
        else:
            y_i.append(y[:,:,k])
    y_i = np.array(y_i)
    y_i_1 = np.ones((np.shape(y)[0], np.shape(y)[1], np.shape(y_i)[0]))
    for k in range(np.shape(y_i)[0]): # final reshaping output array
        y_i_1[:,:,k] = y_i[k,:,:]
    return y_i_1, hist

def count_cat(k):
    """ Counts amount of categories in classification timeseries in column x
    param: k. Column of timeseries. Should be classification data (integer)
    
    returns: amount_cat. Amount of labels in timeseries (integer)
    """
    for n in range(len(out_types)-k): # while search for amount of categories in time series
        # we loop forward and count steps with n
        # as soon as another type of timeseries starts or
        # another classification timeseries starts or
        # something undefined is found
        # the loop stops
        if extract_letter(out_types[k]) in extract_letter(out_types[k+n]):# check for change of type
            # print("No type change")
            try:
                if extract_number(out_types[k+n]) < extract_number(out_types[k+n+1]): # check for start with same type
                    # print("No restart")
                    continue
                else:
                    # print("restart")
                    return extract_number(out_types[k+n])+1
            except:
                # print("No further feature")
                return extract_number(out_types[k+n])+1
        # print("Change of type")
        return extract_number(out_types[k+n-1])+1 # If change of type
    # Ende von out_type hat probleme 63 statt 64
    # vielleicht for loop falsche range

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

class DataGenerator(K.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, X, y, length_item=int(2**9), INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, batch_size=32,
                 shuffle=False):
        "Initialization"
        self.length_item = length_item
        self.INPUT_name = INPUT_name
        self.OUTPUT_name = OUTPUT_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.list_IDs = list_IDs
        self.data_X = X
        self.data_y = y
        self.indexes = np.arange(len(self.data_X[:,0]))
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_X[:,0]) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)
        
        return (X, y)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_X[:,0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.length_item), dtype=float)
        y_ecg = np.empty((self.batch_size, self.length_item), dtype=float)
        y_bbi = np.empty((self.batch_size, len(self.data_y[1][0,:])), dtype=float)

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            # X[i,:] = self.data_X[ID,:]
            X[i,:] = np.load('data/generator/X-'+ str(ID) + ".npy")

            # Store labels
            # y_ecg[i,:] = self.data_y[0][ID,:]
            y_ecg[i,:] = np.load('data/generator/y_ecg-'+ str(ID) + ".npy")
            # y_bbi[i,:] = self.data_y[1][ID,:]
            y_bbi[i,:] = np.load('data/generator/y_bbi-'+ str(ID) + ".npy")

        return X, {"ECG_output": y_ecg, "BBI_output": y_bbi}
    
def calc_symboldynamics(beat_to_beat_intervals, a, mode):
    """ Function to determine symbols for dynamics of beat-to-beat-intervals
    
    a: float. parameters of symbol definition
    mode: mode of differentiation
    
    """