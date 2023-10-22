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
import json
import keras

def train(total_epochs=250, 
          INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, # , "symbols":["lag 0"], "moving average: ["0.25"]
          NNsize=int(2 ** 4), length_item=int(2**9), Arch="Conv-AE-LSTM-P", dataset="SYNECG", weight_check=False, kernel_size=2,
          weight_decay=False, batch_size=10, weight_SymbolC=False, FW_60=False, FW_weight_loss=False, task_loss_MSE=True):
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
    :param weight_check: Sets weights in pseudo loss on or off
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
        mlflow.log_param("Weighted pseudo-Tasks", weight_check)
        mlflow.log_param("kernel_size", kernel_size)
        mlflow.log_param("batch_size", batch_size)
        
        print("\nMemorize data ...")
        
        data_list = list(OUTPUT_name.keys()) + list(INPUT_name.keys()) # list of mentioned features
        data_list = tl.unique(data_list) # sorts out multiple occurences
        
        # different datasets for training and test
        if dataset == "SYNECG": # synthetic ECG from Fabians Bachelor thesis
            data, samplerate = tl.memorize("/mnt/scratchpad/dataOruc/data/training.h5", data_list)    
            data_test, samplerate = tl.memorize("/mnt/scratchpad/dataOruc/data/test.h5", data_list)
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
            with open('/mnt/scratchpad/dataOruc/data/current-set/CONFIG', 'rb') as fp:
                config = json.load(fp)
            amount = int(config["segment_count"]*0.8) # int(150000*0.8) # np.shape(data[list(data.keys())[0]])[0] # amount of training examples
            mlflow.log_param("number of training examples", amount)
            # define outtypes
            # X_test, y_test, patient_ID = DGl.load_chunk_to_variable("Test")
            out_types = DGl.output_type(OUTPUT_name)
        elif dataset == "pretraining":
            length_item = 300 # CVP Datensatz fest gelegte Länge
            samplerate = 256
            chunk_list = [f for f in os.listdir('/mnt/scratchpad/dataOruc/data/pretraining-set/') if (os.path.isfile(os.path.join('/mnt/scratchpad/dataOruc/data/pretraining-set/', f)) and not("patient_id" in f))]
            chunk_list = [int(f[2:-4]) for f in chunk_list if "npy" in f] # list of numbering of chunks
            amount = int(len(chunk_list)*400)
            mlflow.log_param("number of training examples", amount)
            out_types = DGl.output_type(OUTPUT_name)
        else:
            sys.exit("dataset not known")
        
        
        # readjust length_item according to samplerate
        length_item = int(length_item*samplerate) # from seconds to samples
        mlflow.log_param("samplerate", samplerate) # samplerate in Hz
        
        if not(dataset in ["CVP", "pretraining"]):
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
            model, ds_samplerate, latent_dim = tl.setup_Conv_Att_E((length_item,1), NNsize, int(samplerate), out_types, weight_check=weight_check)  # Conv Att Encoder
            mlflow.log_param("Architecture", "Conv_Att_E")  # logs type of architecture
            mlflow.log_param("samperate_in_latent_space", ds_samplerate)  # samplerate after downsampling
            mlflow.log_param("depth_of_latent_space", latent_dim)  # number of feature in latent space
        if Arch == "Conv_Att_E_improved":
            model, ds_samplerate, latent_dim = tl.setup_Conv_Att_E_improved((length_item,1), kernel_size, int(samplerate), out_types, weight_check=weight_check)  # Conv Att Encoder
            mlflow.log_param("Architecture", "Conv_Att_E_improved")  # logs type of architecture
            mlflow.log_param("samperate_in_latent_space", ds_samplerate)  # samplerate after downsampling
            mlflow.log_param("depth_of_latent_space", latent_dim)  # number of feature in latent space
        if Arch == "Conv-LSTM-E":
            # Second Experiment
            model, ds_samplerate = tl.setup_Conv_LSTM_E((length_item,1), NNsize, int(samplerate), out_types)  # Conv Encoder. LSTM+Dense Core + Branch
            mlflow.log_param("Architecture", "Conv-LSTM-E")  # logs type of architecture
            mlflow.log_param("down_samplerate", ds_samplerate)  # samplerate after downsampling
        if Arch == "Conv_Att_E_no_branches":
            model, ds_samplerate, latent_dim = tl.setup_Conv_Att_E_no_branches((length_item,1), NNsize, int(samplerate), out_types, weight_check=weight_check)  # Conv Att Encoder
            mlflow.log_param("Architecture", "Conv_Att_E")  # logs type of architecture
            mlflow.log_param("samperate_in_latent_space", ds_samplerate)  # samplerate after downsampling
            mlflow.log_param("depth_of_latent_space", latent_dim)  # number of feature in latent space
        if Arch == "Conv_Att_E_kernel":
            model, ds_samplerate, latent_dim = tl.setup_Conv_Att_E_kernel((length_item,1), kernel_size, int(samplerate), out_types, weight_check=weight_check)  # Conv Att Encoder
            mlflow.log_param("Architecture", "Conv_Att_E_kernel")  # logs type of architecture
            mlflow.log_param("samperate_in_latent_space", ds_samplerate)  # samplerate after downsampling
            mlflow.log_param("depth_of_latent_space", latent_dim)  # number of feature in latent space
        if Arch == "Conv_Att_E_final":
            model, ds_samplerate = tl.setup_Conv_Att_E_final((length_item,1), kernel_size, int(samplerate), out_types, FW_weight_loss_=FW_weight_loss)  # Conv Att Encoder
            mlflow.log_param("Architecture", "Conv_Att_E_final")  # logs type of architecture
            mlflow.log_param("samperate_in_latent_space", ds_samplerate)  # samplerate after downsampling

        # print(np.shape(X)[1:])
        model.summary()
        tl.draw_model(model)
        mlflow.log_artifact("./model.png")  # links plot to MLFlow run
        mlflow.log_param("number of parameters", model.count_params())  # logs number of parameters
        # exit()
        # Selecting loss and metric functions
        alpha = tf.Variable(1)
        
        if weight_SymbolC:
            loss_SymbolC = ["Symbols_output", tl.symbols_loss_uniform_weighted, 'sparse_categorical_accuracy']
        else:
            loss_SymbolC = ["Symbols_output", tl.symbols_loss_uniform, 'sparse_categorical_accuracy']
        
        if task_loss_MSE:
            loss_task = ["forbword_output", tl.task_loss, 'mean_absolute_percentage_error']
        else:
            loss_task = ["forbword_output", tl.task_loss_MAE, 'mean_absolute_percentage_error']
        
        dic_loss = {
                    "ECG":                  ["ECG_output", tl.pseudo_loss, 'MAE'],
                    "Tacho":                ["Tacho_output", tl.pseudo_loss, 'MAE'],
                    "symbolsC":             loss_SymbolC,
                    "Shannon":              ["Shannon_output", tl.pseudo_loss, 'mean_absolute_percentage_error'], 
                    "Polvar10":             ["Polvar10_output", tl.pseudo_loss, 'mean_absolute_percentage_error'], 
                    "forbword":             loss_task,
                    "SNR":                  ["SNR_output", tl.pseudo_loss, 'MAE'],
                    "words":                ["Words_output", tl.pseudo_loss, 'MAE'],
                    "parameters":           ["parameter_output", tl.pseudo_loss, 'MAE'],
                    "parametersTacho":      ["parameter_output", tl.pseudo_loss, 'MAE'], 
                    "parametersSymbols":    ["parameter_output", tl.pseudo_loss, 'MAE'], 
                    "parametersWords":      ["parameter_output", tl.pseudo_loss, 'MAE']
                    }
        loss = {} # contains loss function with corresponding output layer name
        metrics = {}
        for key in OUTPUT_name: # loop over output features
            loss[dic_loss[key][0]] = dic_loss[key][1]
            metrics[dic_loss[key][0]] = dic_loss[key][2]
        
        # Compile model
        print("\nCompiling model...")
        if weight_decay:
            opt = keras.optimizers.AdamW()
        else:
            opt = 'Adam'
        model.compile(loss= loss,
                    optimizer=opt,
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
        # escb = EarlyStopping(monitor='loss', patience=min(int(total_epochs/5),1), min_delta=1, mode="min", restore_best_weights=True) # loss
        # escb = EarlyStopping(monitor='forbword_output_mean_absolute_percentage_error', patience=min(int(total_epochs/,50), min_delta=0.1, mode="min", restore_best_weights=True) # forbword
        ca = tl.changeAlpha(alpha=alpha)
        
        # Train model on training set
        if not(dataset in ["CVP", "pretraining"]):
            print("\nTraining model...")
            model.fit(X,  # sequence we're using for prediction
                    y,  # sequence we're predicting
                    batch_size= batch_size,# int(np.shape(X)[0] / 2**3), # how many samples to pass to our model at a time
                    # callbacks=[escb], # callback must be in list, otherwise mlflow.autolog() breaks
                    epochs=total_epochs)
        elif dataset == "CVP":
            if FW_60:
                training_generator = DGl.DataGenerator_60(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=batch_size, ToT="Training")
            else:
                training_generator = DGl.DataGenerator(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=batch_size, ToT="Training")
            model.fit(x=training_generator,
                        callbacks=[tf.keras.callbacks.TerminateOnNaN(), ca], # callback must be in list, otherwise mlflow.autolog() breaks
                        epochs=total_epochs
                        )
        elif dataset == "pretraining":
            if FW_60:
                training_generator = DGl.DataGenerator_60(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=batch_size, ToT="pretraining")
            else:
                training_generator = DGl.DataGenerator(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=batch_size, ToT="pretraining")
            model.fit(x=training_generator,
                        callbacks=[tf.keras.callbacks.TerminateOnNaN(), ca], # callback must be in list, otherwise mlflow.autolog() breaks
                        epochs=total_epochs
                        )

def posttrain(ExpID, runID, INPUT_name, OUTPUT_name):
    """trains pretrained models with RX dataset channel 2

    Args:
        runID (_type_): _description_
    """
    mlflow.tensorflow.autolog()
    custom_loss = {"ECG_loss": tl.ECG_loss,
                    "pseudo_loss": tl.pseudo_loss,
                    "task_loss": tl.task_loss,
                    "task_loss_MAE": tl.task_loss_MAE,
                    "symbols_loss_uniform": tl.symbols_loss_uniform,
                    "symbols_loss_uniform_weighted": tl.symbols_loss_uniform_weighted} # Custom loss must be declared for loading model
    
    with mlflow.start_run(runID): # separate run for each NN size and configuration
        with mlflow.start_run(nested=True) as child_run:
            if os.path.exists("mlruns/"+ ExpID +"/"+ runID +"/artifacts/model/data/model"):
                model = tf.keras.models.load_model("mlruns/"+ ExpID +"/"+ runID +"/artifacts/model/data/model", custom_objects=custom_loss)
            else:
                print("No model found in Run. Continue with next Run")
                return

            # Evaluate trained model
            print("\nTraining model...")
            training_generator = DGl.DataGenerator(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=10, ToT="Training")
            model.fit(x=training_generator,
                        callbacks=[tf.keras.callbacks.TerminateOnNaN()], # callback must be in list, otherwise mlflow.autolog() breaks
                        epochs=5
                        )