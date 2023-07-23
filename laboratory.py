"""This is the laboratory
Here we set up loops for training and testing different setups of NN

"""
import mlflow
import os
from train_final import train
# from train_MS import train_MS

# Sets target file for saving runs
# mlflow.set_tracking_uri()

# Zeitverzögerung
# import time
# time.sleep(6300)



# Starting a Run by calling train.py
# we can optionaly pass arguments to train
# all arguments have default values
# the order of the arguments is important
# all arguments before a manually changed argument need to be set
# Arguments
# INPUT_name: Features in Input set
# OUTPUT_name: Features in Output set
# NNsize: width of Input layer of NN
# length_item: number of data points in items

Out_list = [{'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'Shannon': ["lag 0"]}, {'Polvar10': ["lag 0"]}, {'forbword': ["lag 0"]},
            {'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'Tacho': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'SNR': ["lag 0"], 'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'Tacho': ["lag 0"], 'forbword': ["lag 0"]},
            {'symbolsC': ["lag 0"], 'forbword': ["lag 0"]},
            {'SNR': ["lag 0"], 'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'forbword': ["lag 0"]}]

# First Experiment Different Pseudo-Task Combinations
# Set experiment active or create new one
experiment = mlflow.set_experiment("Experiment 1 - Pseudo-Tasks") # Conv-AE-LSTM-P good
print("Experiment_id: {}".format(experiment.experiment_id))

for Out in Out_list:
      train(NNsize=int(2**4), 
            total_epochs=10, 
            length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
            # INPUT_name = {"symbols": ["lag 0"]},
            OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
            Arch = "Conv_Att_E",
            dataset="CVP")# "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")# 
      exit()

# Second Experiment LSTM
experiment = mlflow.set_experiment("Experiment 2 - LSTM") # Conv-AE-LSTM-P good
print("Experiment_id: {}".format(experiment.experiment_id))

train(NNsize=int(2**4), 
      total_epochs=10, 
      length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
      # INPUT_name = {"symbols": ["lag 0"]},
      OUTPUT_name = {'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]}, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
      Arch = "Conv-LSTM-E",
      dataset="CVP")# "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")

# Third Experiment
# Hier weiter coden


# for N in range(16,18,1):
#       print(N)
#       check = "n" # Statement for while-loop, will be set by user
#       while not(check == "y"):
#             train(NNsize=int(2**2), 
#                   total_epochs=50, 
#                   length_item= 2**N, 
#                   # INPUT_name = {"symbols": ["lag 0"]},
#                   OUTPUT_name = {"ECG": ["lag 0"]},# {"BBI":["lag 0"], "symbols": ["lag 0"], "words": ["lag 0"]}, #  "BBI": ["lag 0"]
#                   Arch = "maxKomp-Conv-AE-LSTM-P") # "LSTM-AE")
#             check = input("Type y if LOSS is acceptable \n Press Enter to continue...")
      
"""for n in range(2**4, ):
      for l in range(2**13):
            train(NNsize=int(n), 
                  total_epochs=5, 
                  length_item=int(l), 
                  OUTPUT_name=["lag 0"],# , "moving average", "lag 50"], #, "lag 100", "lag 250", "lag 500"]) # OUTPUT_name=["lag 50"], 
                  Arch = "Conv-AE-LSTM-P")"""