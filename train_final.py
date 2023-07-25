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
import train_lib_final as tl
import attention_lib as al
import matplotlib.pyplot as plt
import numpy as np
import warnings
import mlflow
import os
import sys
from keras.callbacks import EarlyStopping
import tensorflow as tf
import DataGen_lib as DGl

def train(total_epochs=250, 
          INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, # , "symbols":["lag 0"], "moving average: ["0.25"]
          NNsize=int(2 ** 4), length_item=int(2**9), Arch="Conv-AE-LSTM-P", dataset="SYNECG"):
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
        mlflow.log_param("sequence length in s", length_item)
        mlflow.log_param("Input features", INPUT_name)  # Logs configuration of model features
        mlflow.log_param("Output features", OUTPUT_name)  # Logs configuration of model features
        mlflow.log_param("size of NN", NNsize)
        
        print("\nMemorize data ...")
        
        data_list = list(OUTPUT_name.keys()) + list(INPUT_name.keys()) # list of mentioned features
        data_list = tl.unique(data_list) # sorts out multiple occurences
        
        # different datasets for training and test
        if dataset == "SYNECG": # synthetic ECG from Fabians Bachelor thesis
            data, samplerate = tl.memorize("./data/training.h5", data_list)    
            data_test, samplerate = tl.memorize("./data/test.h5", data_list)
        elif dataset == "MIT-BIH": # did not work with LSTM. Never tested with attention
            data, samplerate = tl.memorize_MIT_data(data_list, 'training')
            data_test, samplerate = tl.memorize_MIT_data(data_list, 'test')
        elif dataset == "Icentia":
            amount = 100 # 6000 # number of training patients
            data, data_test, samplerate = tl.Icentia_memorize(amount, length_item, data_list)
            mlflow.log_param("number of training examples", amount)
        elif dataset == "CVP":
            length_item = 300 # CVP Datensatz fest gelegte Länge
            samplerate = 256
            # data, data_test, samplerate = tl.CVP_memorize(data_list)
            amount = 170000 # np.shape(data[list(data.keys())[0]])[0] # amount of training examples
            mlflow.log_param("number of training examples", amount)
            DGl.preprocess(500, INPUT_name, OUTPUT_name) # prepares data for generator
            # define outtypes
            out_types = DGl.output_type(OUTPUT_name)
        else:
            sys.exit("dataset not known")
        
        
        # readjust length_item according to samplerate
        length_item = int(length_item*samplerate) # from seconds to samples
        mlflow.log_param("samplerate", samplerate) # samplerate in Hz
        
        if dataset != "CVP":
            print("\nConstructing training data ...")
            # Extracting items from training data
            X, y, out_types= tl.set_items(data, INPUT_name, OUTPUT_name, length_item)
            print("Types of output: ", out_types)
            tl.feat_check(X,y) # Plots example of dataset and some features
            # exit()
            
            print("\nConstructing test data ...")
            # Extracting items from test data
            X_test, y_test, out_types = tl.set_items(data_test, INPUT_name, OUTPUT_name, length_item)
            # tl.check_data(X,y,X_test,y_test) # Plots diagramm of items and basic values
        
        
        # Initialize model
        print("\nInitializing model...")
        if Arch == "Conv-AE-LSTM-P":
            model, ds_samplerate = tl.setup_Conv_AE_LSTM_P((np.shape(X)[1],1), NNsize, int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv-AE-LSTM-P")  # logs type of architecture
            mlflow.log_param("down_samplerate", ds_samplerate)  # samplerate after downsampling
        if Arch == "Conv_E_LSTM_Att_P":
            model, ds_samplerate = tl.setup_Conv_E_LSTM_Att_P((np.shape(X)[1],1), NNsize, int(samplerate))  # Conv Encoder. LSTM Decoder
            mlflow.log_param("Architecture", "Conv_E_LSTM_Att_P")  # logs type of architecture
            mlflow.log_param("down_samplerate", ds_samplerate)  # samplerate after downsampling
        if Arch == "Conv_Att_E":
            model, ds_samplerate, latent_dim = tl.setup_Conv_Att_E((length_item,1), NNsize, int(samplerate), out_types)  # Conv Att Encoder
            mlflow.log_param("Architecture", "Conv_Att_E")  # logs type of architecture
            mlflow.log_param("samperate_in_latent_space", ds_samplerate)  # samplerate after downsampling
            mlflow.log_param("depth_of_latent_space", latent_dim)  # number of feature in latent space
        if Arch == "Conv-LSTM-E":
            # Second Experiment
            model, ds_samplerate = tl.setup_Conv_LSTM_E((length_item,1), NNsize, int(samplerate), out_types)  # Conv Encoder. LSTM+Dense Core + Branch
            mlflow.log_param("Architecture", "Conv-LSTM-E")  # logs type of architecture
            mlflow.log_param("down_samplerate", ds_samplerate)  # samplerate after downsampling

        # print(np.shape(X)[1:])
        model.summary()
        tl.draw_model(model)
        mlflow.log_artifact("./model.png")  # links plot to MLFlow run
        mlflow.log_param("number of parameters", model.count_params())  # logs number of parameters
        # exit()
        # Selecting loss and metric functions
        alpha = tf.Variable(1)
        dic_loss = {
                    "ECG":                  ["ECG_output", tl.ECG_loss, 'MAE'],
                    "Tacho":                ["Tacho_output", tl.ECG_loss, 'MAE'],
                    "symbolsC":             ["Symbols_output", tl.symbols_loss_uniform, 'sparse_categorical_accuracy'],
                    "Shannon":              ["Shannon_output", tl.ECG_loss, 'mean_absolute_percentage_error'], 
                    "Polvar10":             ["Polvar10_output", tl.ECG_loss, 'mean_absolute_percentage_error'], 
                    "forbword":             ["forbword_output", tl.ECG_loss, 'mean_absolute_percentage_error'],
                    "SNR":                  ["SNR_output", tl.ECG_loss, 'MAE'],
                    "words":                ["Words_output", tl.ECG_loss, 'MAE'],
                    "parameters":           ["parameter_output", tl.ECG_loss, 'MAE'],
                    "parametersTacho":      ["parameter_output", tl.ECG_loss, 'MAE'], 
                    "parametersSymbols":    ["parameter_output", tl.ECG_loss, 'MAE'], 
                    "parametersWords":      ["parameter_output", tl.ECG_loss, 'MAE']
                    }
        loss = {} # contains loss function with corresponding output layer name
        metrics = {}
        for key in OUTPUT_name: # loop over output features
            loss[dic_loss[key][0]] = dic_loss[key][1]
            metrics[dic_loss[key][0]] = dic_loss[key][2]
        
        # Compile model
        print("\nCompiling model...")
        model.compile(loss= loss,
                    optimizer='Adam',
                    metrics=metrics)
        
        # {
        #                 # "Tacho_output": 'MAE',
        #                 # "Symbols_output": 'sparse_categorical_accuracy',
        #                 # "Words_output": 'MAE',
        #                 # "parameter_output": 'MAE',
        #                 "ECG_output": 'MAE',
        #                 }
        # Callback
        # escb = EarlyStopping(monitor='MAE', patience=min(int(total_epochs/5),50), min_delta=0.005, mode="min") # 'Tacho_output_MAE' 'ECG_output_MAE' 'binary_accuracy'
        # escb = EarlyStopping(monitor='sparse_categorical_accuracy', patience=min(int(total_epochs/5),50), min_delta=0.001, mode="max", restore_best_weights=True) # Symbols_output_
        # escb = EarlyStopping(monitor='Symbols_output_sparse_categorical_accuracy', patience=min(int(total_epochs/5),50), min_delta=0.001, mode="max", restore_best_weights=True) # Symbols_output_
        # escb = EarlyStopping(monitor='loss', patience=min(int(total_epochs/5),50), min_delta=1, mode="max", restore_best_weights=True) # loss
        escb = EarlyStopping(monitor='forbword_output_mean_absolute_percentage_error', patience=min(int(total_epochs/2),50), min_delta=0.1, mode="max", restore_best_weights=True) # forbword
        # ca = tl.changeAlpha(alpha=alpha)
        
        # Train model on training set
        if dataset != "CVP":
            print("\nTraining model...")
            model.fit(X,  # sequence we're using for prediction
                    y,  # sequence we're predicting
                    batch_size= 8,# int(np.shape(X)[0] / 2**3), # how many samples to pass to our model at a time
                    # callbacks=[escb], # callback must be in list, otherwise mlflow.autolog() breaks
                    epochs=total_epochs)
        else:
            training_generator = DGl.DataGenerator(INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=8, ToT="Training")
            model.fit_generator(generator=training_generator,
                                # callbacks=[escb], # callback must be in list, otherwise mlflow.autolog() breaks
                                epochs=total_epochs
                                )

        
        # Evaluate trained model
        print("\nEvaluating model...")
        if dataset != "CVP":
            dic_eval = model.evaluate(X_test, y_test, batch_size=8, return_dict=True)
        else:
            dic_eval = model.evaluate(generator=training_generator, return_dict=True)
        
        # Predict on test set and plot
        if dataset != "CVP":
            y_pred = model.predict(X_test, batch_size=16)# int(np.shape(X_test)[0] / 3))
        else:
            y_pred = model.predict(generator=training_generator)
        
        if not(isinstance(y_pred,list)): # check, ob y_pred list ist. Falls mehrere Outputs, dann ja
            y_pred = [y_pred]
        
        # Transform outpur of sparse categorical from 4D time series into 1D time series
        for column in range(len(y_pred)): # loop over features
            if "classificationSymbols" in out_types[column]:
                    # sparse_pred = np.array([0,1,2,3]) * y_pred[column] # weighted sum of labels
                    # sparse_pred = np.sum(sparse_pred, axis=-1)
                    # y_pred[column] = sparse_pred
                    print(np.shape(y_pred[column]))
                    max_pred = np.argmax(y_pred[column], axis=-1)
                    print(np.shape(max_pred))
                    y_pred[column] = max_pred
        
        example = np.random.randint(len(y_test[0][:,0]))
        plt.figure(1)
        plt.title("Full plot Truth and Pred of column 0")
        plt.plot(list(range(len(y_test[0][0,:]))), y_test[0][example,:])
        plt.plot(list(range(len(y_pred[0][0,:]))), y_pred[0][example,:])
        if "ECG" in out_types[0]:
                plt.plot(list(range(len(X_test[0,:]))), X_test[example,:])
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
        
        for k in range(len(y_pred)):
            plt.figure(k)
            if out_types[k] in ["Shannon", "Polvar10", "forbword"]: # Check if column is non-linear parameter
                if out_types[k] in ["Shannon"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                    t = y_test[k][example]*4
                    p = y_pred[k][example]*4
                elif out_types[k] in ["forbword"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                    t = (y_test[k][example]-0.1)*40
                    p = (y_pred[k][example]-0.1)*40
                else:
                    t = y_test[k][example]-0.1
                    p = y_pred[k][example]-0.1
                plt.title("Truth and Pred of column " + str(k) + " rel. error of " + str(abs((t-p)/t)))
                plt.plot(0, t, marker='o')
                plt.plot(0, p, marker='o')
            else: # ECG, Tachogramm oder Symbole
                plt.title(concat("Zoomed Truth and Pred of column ", str(k)))
                plt.plot(list(range(len(y_test[k][0,-2*samplerate:]))), y_test[k][example,-2*samplerate:])
                plt.plot(list(range(len(y_pred[k][0,-2*samplerate:]))), y_pred[k][example,-2*samplerate:])
            if "ECG" in out_types[k]:
                plt.plot(list(range(len(X_test[0,-2*samplerate:]))), X_test[example,-2*samplerate:])
            plt.legend(["y_Test", "Prediction", "X_Test"])
            name = concat("./ZoomPlot-Col-",str(k))
            plt.savefig(name)
            path_fig = "./Prediction-Image/" + str(z)+ "-Zoom-col" + str(k) + ".png"
            plt.savefig(path_fig)  # saves plot
            mlflow.log_artifact(path_fig)  # links plot to MLFlow run
            plt.close()
            
        # Plot multiple examples
        for l in range(10):
            example = np.random.randint(len(y_test[0][:,0]))
            for k in range(len(y_pred)):
                # Full visuzilation
                plt.figure(k)
                if out_types[k] in ["Shannon", "Polvar10", "forbword"]: # Check if column is non-linear parameter
                    if out_types[k] in ["Shannon"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                        t = y_test[k][example]*4
                        p = y_pred[k][example]*4
                    elif out_types[k] in ["forbword"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                        t = (y_test[k][example]-0.1)*40
                        p = (y_pred[k][example]-0.1)*40
                    else:
                        t = y_test[k][example]-0.1
                        p = y_pred[k][example]-0.1
                    plt.title("Truth and Pred of column " + str(k) + " of example " + str(l) + " rel. error of " + str(np.round(abs((t-p)/t))[0]))
                    plt.plot(0, t, marker='o')
                    plt.plot(0, p, marker='o')
                else: # ECG, Tachogramm oder Symbole
                    plt.title(concat("Truth and Pred of column " + str(k), " of example " + str(l)))
                    plt.plot(list(range(len(y_test[k][0,:]))), y_test[k][example,:])
                    plt.plot(list(range(len(y_pred[k][0,:]))), y_pred[k][example,:])
                if "ECG" in out_types[k]:
                    plt.plot(list(range(len(X_test[0,:]))), X_test[example,:])
                plt.legend(["y_Test", "Prediction", "X_Test"])
                name = "./plots/Plot-Col-" + str(k) + "-example-" + str(l)
                plt.savefig(name)
                plt.close()
                # Zoom visualization
                plt.figure(k)
                if out_types[k] in ["Shannon", "Polvar10", "forbword"]: # Check if column is non-linear parameter
                    if out_types[k] in ["Shannon"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                        t = y_test[k][example]*4
                        p = y_pred[k][example]*4
                    elif out_types[k] in ["forbword"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                        t = (y_test[k][example]-0.1)*40
                        p = (y_pred[k][example]-0.1)*40
                    else:
                        t = y_test[k][example]-0.1
                        p = y_pred[k][example]-0.1
                    plt.title("Truth and Pred of column " + str(k) + " of example " + str(l) + " rel. error of " + str(np.round(abs((y_test[k][example]-y_pred[k][example])/y_test[k][example]), decimals=2)[0]))
                    plt.plot(0, t, marker='o')
                    plt.plot(0, p, marker='o')
                else: # ECG, Tachogramm oder Symbole
                    plt.title(concat("Truth and Pred of column " + str(k), " of example " + str(l)))
                    plt.plot(list(range(len(y_test[k][0,-2*samplerate:]))), y_test[k][example,-2*samplerate:])
                    plt.plot(list(range(len(y_pred[k][0,-2*samplerate:]))), y_pred[k][example,-2*samplerate:])
                if "ECG" in out_types[k]:
                    plt.plot(list(range(len(X_test[0,-2*samplerate:]))), X_test[example,-2*samplerate:])
                plt.legend(["y_Test", "Prediction", "X_Test"])
                name = "./plots/Plot-Col-" + str(k) + "-example-Zoom-" + str(l)
                plt.savefig(name)
                plt.close()
                
            # plot Input
            plt.figure(2)
            plt.title(concat("Input of example ", str(l)))
            plt.plot(list(range(len(X_test[0,:]))), X_test[example,:])
            name = "./plots/Plot-Input-example-" + str(l)
            plt.savefig(name)
            plt.close()
            
            # plot parameters of all examples
            for k in range(len(y_pred)):
                if out_types[k] in ["Shannon", "Polvar10", "forbword"]:
                    if out_types[k] in ["Shannon"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                        t = y_test[k][:]*4
                        p = y_pred[k][:]*4
                    elif out_types[k] in ["forbword"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                        t = (y_test[k][:]-0.1)*40
                        p = (y_pred[k][:]-0.1)*40
                    else:
                        t = y_test[k][:]-0.1
                        p = y_pred[k][:]-0.1
                    re = np.round(np.mean(abs((y_test[k][:]-y_pred[k][:])/y_test[k][:])), decimals=2)
                    plt.figure(1)
                    plt.title("Parameter " + out_types[k] + " of all examples with rel. error " + str(re)) # rel. error einfügen
                    plt.plot(list(range(len(t))), t)
                    plt.plot(list(range(len(p))), p)
                    plt.legend(["y_Test", "Prediction"])
                    name = "./plots/data-Col-" + out_types[k] + "-all-examples"
                    plt.savefig(name)
                    z = np.random.randint(100000, 999999)
                    path_fig = "./Prediction-Image/" + str(z)+ "-data-" + out_types[k] + "-all-examples.png"
                    plt.savefig(path_fig)  # saves plot
                    mlflow.log_artifact(path_fig)  # links plot to MLFlow run
                    plt.close()