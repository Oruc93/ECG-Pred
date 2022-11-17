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

# OUTPUT_name = ["lag 1"]# , "lag 5", "lag 11", "moving average"]
# print(len(OUTPUT_name))

"""data = tl.memorize("./data/5min.h5", 'ECG')
print(np.shape(data))
plt.figure(1)
plt.title("ECG with 5 minute duration")
plt.plot(list(range(len(data[0,0:2000]))), data[0,0:2000])
plt.savefig("Test.png")
"""
def train(total_epochs=250, 
          INPUT_name=["lag 0"], OUTPUT_name=["lag 1", "lag 100", "lag 200", "moving average"],
          NNsize=int(2 ** 7), length_item=int(10000), Arch="Autoencoder-LSTM", samplerate = 1000):
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
        mlflow.log_param("width of input layer", NNsize)
        
        print("\nMemorize training data ...")
        # Extracting dataset from h5 into numpy array
        data, orig_samplerate = tl.memorize("./data/5min.h5", 'ECG_training')
        print("Constructing training data ...")
        # Extracting items from whole dataset
        X, y = tl.set_items(data, INPUT_name, OUTPUT_name, length_item, orig_samplerate)
        
        # X = tl.downsample(X, orig_samplerate, samplerate)
        # y = tl.downsample(y, orig_samplerate, samplerate)
        
        plt.figure(1)
        print("Shape of X", np.shape(X[0,-2*samplerate:,0]))
        plt.plot(list(range(len(X[0,-2*samplerate:,0]))), X[0,-2*samplerate:,0])
        plt.title("Convolution with 2s-Kernels so they will look like this")
        plt.savefig("Test.png")
        np.save("2s-snippet", X[0,-2*samplerate:,0])
        
        tl.feat_check(X,y) # Plots example of dataset and some features
        
        print("\nMemorize test data ...")
        # Extracting dataset from h5 into numpy array
        data, orig_samplerate = tl.memorize("./data/5min.h5", 'ECG_test')
        print("Constructing test data ...")
        # Extracting items from whole dataset
        X_test, y_test = tl.set_items(data, INPUT_name, OUTPUT_name, length_item, orig_samplerate)
        
        # X_test = tl.downsample(X_test, orig_samplerate, samplerate)
        # y_test = tl.downsample(y_test, orig_samplerate, samplerate)
        
        tl.check_data(X,y,X_test,y_test) # Plots diagramm of items and basic values
        mlflow.log_param("orig_samplerate", orig_samplerate) # samplerate in Hz
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
            model = tl.setup_Conv_AE_LSTM_P(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(orig_samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-AE-LSTM-P")  # logs type of architecture
        elif Arch == "Conv-AE":
            model = tl.setup_Conv_AE(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(orig_samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-AE")  # logs type of architecture
        elif Arch == "Conv-AE-Dense":
            model = tl.setup_Conv_AE_Dense(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(orig_samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-AE-Dense")  # logs type of architecture
        """elif Arch == "Conv-E-LSTM-P":
            model, model_latent = tl.setup_Conv_DS_Dense_E_LSTM_P(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(orig_samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-E-LSTM-P")  # logs type of architecture"""
        """elif Arch == "Conv_DS_Dense_E_LSTM_P":
            model, encoder = tl.setup_Conv_DS_Dense_E_LSTM_P(np.shape(X)[1:], NNsize, np.shape(y)[-1], int(orig_samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv_DS_Dense_E_LSTM_P")  # logs type of architecture"""
        
        # print(np.shape(X)[1:])
        model.summary()
        # exit()
        tl.draw_model(model)
        mlflow.log_param("number of parameters", model.count_params())  # logs number of parameters
        
        # Compile model
        print("Compiling model...")
        model.compile(loss= tl.my_loss_fn, # Loss-Funktion
                    optimizer='Adam',
                    metrics='MAE')
        
        # Train model on training set
        print("Training model...")
        model.fit(X,  # sequence we're using for prediction
                y,  # sequence we're predicting
                batch_size=int(np.shape(X)[0] / 10), # how many samples to pass to our model at a time
                callbacks=EarlyStopping(monitor="MAE",patience=7),
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
        if len(np.shape(y_pred))==4: # if model output has 4 dimension. Correct it to 3
            y_pred = y_pred[0,:,:,:]
            print(np.shape(y_pred))
        print(np.shape(y_test))
        
        """np.save("./X_test", X_test) # Saving output in a file for later use
        np.save("./y_test", y_test)
        np.save("./y_pred", y_pred)"""
        
        plt.figure(1)
        plt.title("Full plot Truth and Pred of column 0")
        plt.plot(list(range(len(y_test[0,:,0]))), y_test[0,:,0])
        plt.plot(list(range(len(y_pred[0,:,0]))), y_pred[0,:,0])
        plt.plot(list(range(len(X_test[0,:,0]))), X_test[0,:,0])
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
                k = 101
            if k == 100:
                print("Could not find image name. Raise limit of rand.int above")
        plt.close()
        
        for k in range(len(y_pred[0,0,:])):
            plt.figure(k+2)
            plt.title(concat("Zoomed Truth and Pred of column ", str(k)))
            plt.plot(list(range(len(y_test[0,-2*samplerate:,k]))), y_test[0,-2*samplerate:,k])
            plt.plot(list(range(len(y_pred[0,-2*samplerate:,k]))), y_pred[0,-2*samplerate:,k])
            plt.plot(list(range(len(X_test[0,-2*samplerate:]))), X_test[0,-2*samplerate:])
            plt.legend(["y_Test", "Prediction", "X_Test"])
            name = concat("./ZoomPlot-Col-",str(k))
            plt.savefig(name)
            path_fig = "./Prediction-Image/" + str(z)+ "-Zoom-col" + str(k) + ".png"
            plt.savefig(path_fig)  # saves plot
            mlflow.log_artifact(path_fig)  # links plot to MLFlow run
            plt.close()
            
            # check if the features are correct
            # efficiency check
            # maybe change data to BBI (less complex data) and 4 Hertz(most cummon)