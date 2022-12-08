# This script contains the Training and Evaluation of one setup of a neural network

# We want to work with Jax and Flax
# Check if MLFlow can work with these frameworks !!!!!!!!!!!
# - Exporting to Tensorflow’s SavedModel with jax2tf
# - JAX released an experimental converter called jax2tf, which allows converting trained Flax models into Tensorflow’s SavedModel format (so it can be used for TF Hub, TF.lite, TF.js, or other downstream applications). The repository contains more documentation and has various examples for Flax.
# If yes, use new conda environment FLAX
# First build model with keras and test a model with FLAX later
# Dominiks test mit FLAX lief nicht gut

# Take train of Temp as template

from operator import concat
import train_lib as tl
import matplotlib.pyplot as plt
import numpy as np
import warnings
import mlflow
import os
from keras.callbacks import EarlyStopping

def train(total_epochs=250, 
          INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, # , "symbols":["lag 0"], "moving average: ["0.25"]
          NNsize=int(2 ** 4), length_item=int(2**9), Arch="Conv-AE-LSTM-P"):
    """
    Setting up, training and evaluating one neural network with specific configuration
    During this process different metrics are logged in MLFlow and can be accessed by MLFlow UI in the browser

    :param total_epochs: number of epochs in training
    :param INPUT_name: selection of input features
    :param OUTPUT_name: selection of output features
    :param NNsize: width of the input layer loosely corresponding to network size
    :param length_item: length of items given into the neural network
    :param Arch: type of neural network architecture
    :param samplerate: goal samplerate to downsample to
    :return:
    """
    warnings.filterwarnings("ignore")
    
    with mlflow.start_run():  # separate run for each NN size and configuration
        
        mlflow.tensorflow.autolog()  # automatically starts logging LOSS, metric and models
        # Setting Parameters of Run
        print("Setting Parameters for Run...")
        # Length of sequence
        mlflow.log_param("sequence length", length_item)
        mlflow.log_param("Input features", INPUT_name)  # Logs configuration of model features
        mlflow.log_param("Output features", OUTPUT_name)  # Logs configuration of model features
        mlflow.log_param("size of NN", NNsize)
        
        print("\nMemorize training data ...")
        # Extracting dataset from h5 into numpy array
        data_list = list(OUTPUT_name.keys()) + list(INPUT_name.keys()) # list of mentioned datasets
        data_list = tl.unique(data_list) # sorts out multiple occurences
        data, samplerate = tl.memorize("./data/training.h5", data_list)
        
        print("Constructing training data ...")
        
        # !!! We need to readjust how item length is defined and time series is cut
        # BBI and ECG have different x-axis
        # maybe length_item is time in s and we transform into indices
        
        # Extracting items from whole dataset
        X, y, out_types= tl.set_items(data, INPUT_name, OUTPUT_name, length_item)
        print(out_types)
        
        tl.kernel_check(X) # Plots 2s snippet of ecg
        
        tl.feat_check(X,y) # Plots example of dataset and some features
        
        print("\nMemorize test data ...")
        # Extracting dataset from h5 into numpy array
        data, samplerate = tl.memorize("./data/test.h5", data_list)
        print("Constructing test data ...")
        # Extracting items from whole dataset
        X_test, y_test, out_types = tl.set_items(data, INPUT_name, OUTPUT_name, length_item)
        
        tl.check_data(X,y,X_test,y_test) # Plots diagramm of items and basic values
        mlflow.log_param("samplerate", samplerate) # samplerate in Hz
        
        # Initialize model
        print("Initializing model...")
        if Arch == "LSTM-AE":
            model = tl.setup_LSTM_AE(np.shape(X)[1:], NNsize, np.shape(y)[-1]) # Autoencoder branched
            mlflow.log_param("Architecture", "Autoencoder-LSTM")  # logs type of architecture
        elif Arch == "LSTM":
            model = tl.setup_LSTM_nn(np.shape(X)[1:], NNsize, np.shape(y)[-1])  # LSTM
            mlflow.log_param("Architecture", "LSTM")  # logs type of architecture
        elif Arch == "Conv-AE-LSTM-P":
            model = tl.setup_Conv_AE_LSTM_P(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-AE-LSTM-P")  # logs type of architecture
        elif Arch == "maxKomp-Conv-AE-LSTM-P":
            model, ds_samplerate = tl.setup_maxKomp_Conv_AE_LSTM_P(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "maxKomp-Conv-AE-LSTM-P")  # logs type of architecture
            mlflow.log_param("down_samplerate", ds_samplerate)  # samplerate after downsampling
        elif Arch == "Conv-AE":
            model = tl.setup_Conv_AE(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-AE")  # logs type of architecture
        elif Arch == "Conv-AE-Dense":
            model = tl.setup_Conv_AE_Dense(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-AE-Dense")  # logs type of architecture
        """elif Arch == "Conv-E-LSTM-P":
            model, model_latent = tl.setup_Conv_DS_Dense_E_LSTM_P(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-E-LSTM-P")  # logs type of architecture"""
        """elif Arch == "Conv_DS_Dense_E_LSTM_P":
            model, encoder = tl.setup_Conv_DS_Dense_E_LSTM_P(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv_DS_Dense_E_LSTM_P")  # logs type of architecture"""
        
        # print(np.shape(X)[1:])
        model.summary()
        tl.draw_model(model)
        mlflow.log_artifact("./model.png")  # links plot to MLFlow run
        mlflow.log_param("number of parameters", model.count_params())  # logs number of parameters
        
        # Compile model
        print("Compiling model...")
        model.compile(loss= tl.my_loss_fn, # Loss-Funktion
                    optimizer='Adam',
                    metrics='MAE')
        
        # Callback
        escb = EarlyStopping(monitor='MAE', patience=int(total_epochs/2), min_delta=0.001, mode="min")
        
        # Train model on training set
        print("Training model...")
        model.fit(X,  # sequence we're using for prediction
                  y,  # sequence we're predicting
                batch_size=int(np.shape(X)[0] / 10), # how many samples to pass to our model at a time
                callbacks=[escb], # callback must be in list, otherwise mlflow.autolog() breaks
                epochs=total_epochs)
        
        # Evaluate trained model
        print("Evaluating model...")
        model.evaluate(X_test, y_test, batch_size=int(np.shape(X)[0] / 10))
        
        # Predict on test set and plot
        y_pred = model.predict(X_test, batch_size=int(np.shape(X)[0] / 10))
        # print(type(y_pred))
        # y_pred = np.array(y_pred) # transform 
        # print(y_pred)
        # print(np.shape(y_pred))
        """if len(np.shape(y_pred))==4: # if model output has 4 dimension. Correct it to 3
            y_pred = y_pred[0,:,:,:]
            print(np.shape(y_pred))"""
        print(np.shape(y_test))
        
        """np.save("./X_test", X_test) # Saving output in a file for later use
        np.save("./y_test", y_test)
        np.save("./y_pred", y_pred)"""
        
        example = np.random.randint(len(y_test[:,0,0]))
        plt.figure(1)
        plt.title("Full plot Truth and Pred of column 0")
        plt.plot(list(range(len(y_test[0,:,0]))), y_test[example,:,0])
        plt.plot(list(range(len(y_pred[0,:,0]))), y_pred[example,:,0])
        plt.plot(list(range(len(X_test[0,:,0]))), X_test[example,:,0])
        plt.legend(["y_test", "y_pred", "X_test"])
        plt.savefig("Full-Plot col 0")
        # while loop for saving image
        k = int(0)
        while k<100:
            k += 1
            z = np.random.randint(100000, 999999)
            path_fig = "./Prediction-Image/" + str(z) + ".png"
            if not os.path.isfile(path_fig):  # checks if file already exists
                plt.savefig(path_fig)  # saves plot
                mlflow.log_artifact(path_fig)  # links plot to MLFlow run
                break
            if k == 100:
                print("Could not find image name. Raise limit of rand.int above")
        plt.close()
        
        # Preparation of output if classification of symbols and words was done
        if "classification" in out_types or "dsitribution" in out_types:
            y_pred, dist_pred = tl.index(y_pred)
            y_test, dist_test = tl.index(y_test)
            print("Shape of final Output array: ", np.shape(y_test))
            
        for k in range(len(y_pred[0,0,:])):
            plt.figure(k+2)
            plt.title(concat("Zoomed Truth and Pred of column ", str(k)))
            plt.plot(list(range(len(y_test[0,-2*samplerate:,k]))), y_test[example,-2*samplerate:,k])
            plt.plot(list(range(len(y_pred[0,-2*samplerate:,k]))), y_pred[example,-2*samplerate:,k])
            plt.plot(list(range(len(X_test[0,-2*samplerate:]))), X_test[example,-2*samplerate:])
            plt.legend(["y_Test", "Prediction", "X_Test"])
            name = concat("./ZoomPlot-Col-",str(k))
            plt.savefig(name)
            path_fig = "./Prediction-Image/" + str(z)+ "-Zoom-col" + str(k) + ".png"
            plt.savefig(path_fig)  # saves plot
            mlflow.log_artifact(path_fig)  # links plot to MLFlow run
            plt.close()
        
        # Plot of classification of symbols and words output
        if "classification" in out_types or "dsitribution" in out_types:
            print("Shape of hist", np.shape(dist_pred))
            # in our case we have four symbols in words of three
            plt.figure(1)
            plt.title(str("Zoomed Truth and Pred of column distribution of " + data_list[k+1]))
            plt.bar(np.arange(len(dist_test[0,:]))-0.5, dist_test[example,:])
            plt.bar(np.arange(len(dist_pred[0,:]))+0.5, dist_pred[example,:])
            # plt.plot(list(range(len(X_test[0,:,0]))), X_test[0,:,0])
            plt.legend(["y_Test", "Prediction", "X_Test 0"])
            name = concat("./ZoomPlot-Col-",str(k+1))
            plt.savefig(name)
            path_fig = "./Prediction-Image/" + str(z)+ "-Zoom-col" + str(k+1) + ".png"
            plt.savefig(path_fig)  # saves plot
            mlflow.log_artifact(path_fig)  # links plot to MLFlow run
            plt.close()
            
        
        