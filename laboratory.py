"""This is the laboratory
Here we set up loops for training and testing different setups of NN

"""
import mlflow
import os
from train import train
from train_MS import train_MS

# Sets target file for saving runs
# mlflow.set_tracking_uri()

# Set experiment active or create new one
experiment = mlflow.set_experiment("Test-1")
print("Experiment_id: {}".format(experiment.experiment_id))

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

train(NNsize=int(2**5), 
      total_epochs=50, 
      length_item= int(2**11), 
      OUTPUT_name=["lag 0", "lag 250"],# , "moving average", "lag 50"], #, "lag 100", "lag 250", "lag 500"]) # OUTPUT_name=["lag 50"], 
      Arch = "Conv-AE-LSTM-P") # "LSTM-AE")
      
"""for n in range(2**4, ):
      for l in range(2**13):
            train(NNsize=int(n), 
                  total_epochs=5, 
                  length_item=int(l), 
                  OUTPUT_name=["lag 0"],# , "moving average", "lag 50"], #, "lag 100", "lag 250", "lag 500"]) # OUTPUT_name=["lag 50"], 
                  Arch = "Conv-AE-LSTM-P")"""