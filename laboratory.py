"""This is the laboratory
Here we set up loops for training and testing different setups of NN

"""
import mlflow
import os
from train_final import train, posttrain
from load import load_proof, load_test
import DataGen_lib as DGl
# from train_MS import train_MS

# Sets target file for saving runs
# mlflow.set_tracking_uri()

# Zeitverzögerung
# import time
# time.sleep(3600)


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
            {'Shannon': ["lag 0"]}, 
            {'Polvar10': ["lag 0"]}, 
            {'forbword': ["lag 0"]},
            {'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'Tacho': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'SNR': ["lag 0"], 'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'Tacho': ["lag 0"], 'forbword': ["lag 0"]},
            {'symbolsC': ["lag 0"], 'forbword': ["lag 0"]},
            # {'SNR': ["lag 0"], 'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'forbword': ["lag 0"]},
            {'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'forbword': ["lag 0"]},
            {'SNR': ["lag 0"], 'forbword': ["lag 0"]}]

# import multiprocessing as mp
# import numpy as np
# import time
# print("number of CPUs ", mp.cpu_count())
# tic = time.time()
# with mp.Pool(3) as pool:
#       pool.map(DGl.preprocess, ["Training", "Test", "Proof"])
#       print("Sekunden für Batch-Processing", np.round(time.time()-tic,3))
# DGl.preprocess(None) # prepares data for generator. Only use if new dataset is used

################################################################################################################
# First Experiment Different Pseudo-Task Combinations
# Set experiment active or create new one

# # load(NNsize=int(2**4), 
# #       total_epochs=10, 
# #       length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
# #       # INPUT_name = {"symbols": ["lag 0"]},
# #       OUTPUT_name = {'forbword': ["lag 0"]}, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
# #       Arch = "Conv_Att_E_improved",
# #       dataset="CVP")

# experiment = mlflow.set_experiment("Experiment 1 - Pseudo-Tasks - 80:20:dNC") # Conv-AE-LSTM-P good
# print("Experiment_id: {}".format(experiment.experiment_id))

# load_test(experiment.experiment_id)
# load_proof(experiment.experiment_id)

# for Out in Out_list[1:]:
#       print(Out)
#       train(NNsize=int(2**4), 
#             total_epochs=10, 
#             length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
#             # INPUT_name = {"symbols": ["lag 0"]},
#             OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
#             Arch = "Conv_Att_E_improved",
#             dataset="CVP")# "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")# 

################################################################################################################
# Second Experiment LSTM
# experiment = mlflow.set_experiment("Experiment 2 - LSTM - 80:20:dNC") # Conv-AE-LSTM-P good
# print("Experiment_id: {}".format(experiment.experiment_id))

# load_test(experiment.experiment_id)
# load_proof(experiment.experiment_id)

# for Out in Out_list:
#       print(Out)
#       train(NNsize=int(2**4), 
#             total_epochs=10, 
#             length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
#             # INPUT_name = {"symbols": ["lag 0"]},
#             OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
#             Arch = "Conv-LSTM-E",
#             dataset="CVP")# "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")

################################################################################################################
# Third Experiment Weighted Pseudo-Tasks
# experiment = mlflow.set_experiment("Experiment 3 - Weighted Pseudo-Tasks - 80:20:dNC") # Conv-AE-LSTM-P good
# print("Experiment_id: {}".format(experiment.experiment_id))

# load_test(experiment.experiment_id)
# load_proof(experiment.experiment_id)

# for Out in Out_list:
#       if 'forbword' in list(Out.keys()):
#             print(Out)
#             train(NNsize=int(2**4), 
#                   total_epochs=10, 
#                   length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
#                   # INPUT_name = {"symbols": ["lag 0"]},
#                   OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
#                   Arch = "Conv_Att_E",
#                   dataset="CVP",
#                   weight_check=True)# "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")

################################################################################################################
# Fourth Experiment Unbranched Network
# experiment = mlflow.set_experiment("Experiment 4 - Unbranched Network - 80:20:dNC") # Conv-AE-LSTM-P good
# print("Experiment_id: {}".format(experiment.experiment_id))

# for Out in Out_list:
#       if 'forbword' in list(Out.keys()):
#             print(Out)
#             train(NNsize=int(2**4), 
#                   total_epochs=10, 
#                   length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
#                   # INPUT_name = {"symbols": ["lag 0"]},
#                   OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
#                   Arch = "Conv_Att_E_no_branches",
#                   dataset="CVP",
#                   weight_check=False)# "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")

# load_test(experiment.experiment_id)
# load_proof(experiment.experiment_id)

#######################################################################################################

# Fifth Experiment Improved Architecture. Escpecially Encoder
# Set experiment active or create new one

Out_list_ = [{'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'forbword': ["lag 0"]},
            {'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'Tacho': ["lag 0"], 'forbword': ["lag 0"]},
            {'symbolsC': ["lag 0"], 'forbword': ["lag 0"]},
            # {'SNR': ["lag 0"], 'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]},
            {'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'forbword': ["lag 0"]}
            # {'SNR': ["lag 0"], 'forbword': ["lag 0"]}
            ]

"""experiment = mlflow.set_experiment("Experiment 5.3 - Architecture Encoder dilation only first Conv- 80:20:dNC") # Conv-AE-LSTM-P good
print("Experiment_id: {}".format(experiment.experiment_id))

for kernel_size in [0.25, 0.5, 1]:
      for Out in Out_list_:
            print(Out)
            print(kernel_size)
            train(NNsize=int(2**4), 
            total_epochs=5, 
            length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
            # INPUT_name = {"symbols": ["lag 0"]},
            OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
            Arch = "Conv_Att_E_improved",
            dataset="CVP",
            kernel_size=kernel_size) # "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")# 

load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)"""



experiment = mlflow.set_experiment("Experiment 5.4 - Architecture Encoder filling haps in 5 - 80:20:dNC") # Conv-AE-LSTM-P good
print("Experiment_id: {}".format(experiment.experiment_id))

# 0.25 T,S,FW
# 0.25 T,FW

"""for kernel_size in [0.25]:
      for Out in [{'Tacho': ["lag 0"], 'forbword': ["lag 0"]},
                  {'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'forbword': ["lag 0"]}]:
            print(Out)
            print(kernel_size)
            train(NNsize=int(2**4), 
            total_epochs=5, 
            length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
            # INPUT_name = {"symbols": ["lag 0"]},
            OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
            Arch = "Conv_Att_E_kernel",
            dataset="CVP",
            kernel_size=kernel_size) # "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")# 

load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)"""

#######################################################################################################

# Sixth Experiment Adam with weight decay
# Set experiment active or create new one

"""experiment = mlflow.set_experiment("Experiment 6.2 - AdamW filling gaps - 80:20:dNC") # Conv-AE-LSTM-P good
print("Experiment_id: {}".format(experiment.experiment_id))

for Out in [{'Tacho': ["lag 0"], 'forbword': ["lag 0"]},
            {'symbolsC': ["lag 0"], 'forbword': ["lag 0"]}
            ]:
      print(Out)
      train(NNsize=int(2**4), 
      total_epochs=5, 
      length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
      # INPUT_name = {"symbols": ["lag 0"]},
      OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
      Arch = "Conv_Att_E_improved",
      dataset="CVP",
      kernel_size=0.25,
      weight_decay=True)

load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)"""

#################################################################################################
# Seventh Experiment Channel 2
# preprocess is ready
# DGl.preprocess("Proof")

"""experiment = mlflow.set_experiment("Experiment 7.2 - Channel 2 filling gaps - 80:20:dNC") 
print("Experiment_id: {}".format(experiment.experiment_id))

for Out in [{'symbolsC': ["lag 0"], 'forbword': ["lag 0"]}]:
      print(Out)
      train(NNsize=int(2**4), 
      total_epochs=5, 
      length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
      # INPUT_name = {"symbols": ["lag 0"]},
      OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
      Arch = "Conv_Att_E_improved",
      dataset="CVP",
      kernel_size=0.25,
      weight_decay=True,
      batch_size=10)

load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)"""

#################################################################################################
# Eightth Experiment Weighted SymbolC
# Loss of SymbolC ~0.7 and rest ~0.007

"""experiment = mlflow.set_experiment("Experiment 8.1 - Weighted SymbolC - 80:20:dNC") 
print("Experiment_id: {}".format(experiment.experiment_id))

for Out in [{'forbword': ["lag 0"]}]:
      print(Out)
      train(NNsize=int(2**4), 
      total_epochs=5, 
      length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
      # INPUT_name = {"symbols": ["lag 0"]},
      OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
      Arch = "Conv_Att_E_improved",
      dataset="CVP",
      kernel_size=0.25,
      weight_decay=True,
      batch_size=10,
      weight_SymbolC=True)

load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)"""
#################################################################################################
# Final Testing
""" - no SNR
    - Attention
    - no Weighted Pseudo-Loss
    - verzweigte Architektur
    - 0,5 Kernel
    - AdamW
    - Channel 2
    - Weighted SymbolC
"""
experiment = mlflow.set_experiment("Experiment 10.1 - Final Testing filling gaps in 10.0- 80:20:dNC") 
print("Experiment_id: {}".format(experiment.experiment_id))

"""for Out in Out_list_:
            print(Out)
            train(
            total_epochs=5, 
            length_item= 300,
            OUTPUT_name = Out,
            Arch = "Conv_Att_E_final",
            dataset="CVP",
            kernel_size=0.5,
            weight_decay=True,
            weight_SymbolC=True)"""

# train
"""ExpID = experiment.experiment_id
INPUT_name={"ECG":["lag 0"]}
runs = mlflow.search_runs(ExpID, run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
print(runs)
for n in range(len(runs)):
      runID = runs["run_id"][n]
      run_name = runs["tags.mlflow.runName"][n]
      OUTPUT_name = eval(runs["params.Output features"][n])
      print("Evaluating model ", run_name, " on test data in Exp ", ExpID)
      out_types = DGl.output_type(OUTPUT_name)
      print(out_types)
      if not("forbword" in out_types): # check if forbword is in Output
            print("No forbword in this model. Continue with next model")
            continue
      posttrain(ExpID, runID, INPUT_name, OUTPUT_name)"""

# load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)

experiment = mlflow.set_experiment("Experiment 10 - Final Testing - 80:20:dNC") 
print("Experiment_id: {}".format(experiment.experiment_id))

# train
"""ExpID = experiment.experiment_id
INPUT_name={"ECG":["lag 0"]}
runs = mlflow.search_runs(ExpID, run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
print(runs)
for n in range(len(runs)):
      runID = runs["run_id"][n]
      run_name = runs["tags.mlflow.runName"][n]
      OUTPUT_name = eval(runs["params.Output features"][n])
      print("Evaluating model ", run_name, " on test data in Exp ", ExpID)
      out_types = DGl.output_type(OUTPUT_name)
      print(out_types)
      if not("forbword" in out_types): # check if forbword is in Output
            print("No forbword in this model. Continue with next model")
            continue
      posttrain(ExpID, runID, INPUT_name, OUTPUT_name)"""

# load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)

###################################################################################
# Eleventh Experiment
# Renormalized FW

"""experiment = mlflow.set_experiment("Experiment 11 - FW divided by 60 - 80:20:dNC") 
print("Experiment_id: {}".format(experiment.experiment_id))

for Out in Out_list_:
            print(Out)
            train(
            total_epochs=5, 
            length_item= 300,
            OUTPUT_name = Out,
            Arch = "Conv_Att_E_final",
            dataset="CVP",
            kernel_size=0.5,
            weight_decay=True,
            weight_SymbolC=True,
            FW_60=True)

load_test(experiment.experiment_id, FW_60=True)
load_proof(experiment.experiment_id, FW_60=True)"""

#################################################################################################
# Fully ready
# Needs testing
# DGl.pretraining_preprocess() # prepares data for generator. Only use if new dataset is used
"""DGl.pretraining_preprocess_Icentia()

# Ninth Experiment Pretraining
experiment = mlflow.set_experiment("Experiment 9 - Pretrained Network - 80:20:dNC") # Conv-AE-LSTM-P good
print("Experiment_id: {}".format(experiment.experiment_id))

# pretrain
for Out in Out_list_:
      if 'forbword' in list(Out.keys()):
            print(Out)
            train(NNsize=int(2**4), 
                  total_epochs=5, 
                  length_item= 300,# 256 2**6, # Minimum 4 seconds. Because calc_symbols needs at leat 2 beats. in seconds
                  # INPUT_name = {"symbols": ["lag 0"]},
                  OUTPUT_name = Out, # 'parametersTacho': ["lag 0"]},# 'symbolsC': ["lag 0"], "words": ["lag 0"]}, "ECG": ["lag 0"], 'Tacho': ["lag 0"]
                  Arch = "Conv_Att_E_improved",
                  dataset="pretraining"
                  )# "Conv_E_LSTM_Att_P") #"Conv-AE-LSTM-P")# "maxKomp-Conv-AE-LSTM-P")

# train
ExpID = experiment.experiment_id
INPUT_name={"ECG":["lag 0"]}
runs = mlflow.search_runs(ExpID, run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
print(runs)
for n in range(len(runs)):
      runID = runs["run_id"][n]
      run_name = runs["tags.mlflow.runName"][n]
      OUTPUT_name = eval(runs["params.Output features"][n])
      print("Evaluating model ", run_name, " on test data in Exp ", ExpID)
      out_types = DGl.output_type(OUTPUT_name)
      print(out_types)
      if not("forbword" in out_types): # check if forbword is in Output
            print("No forbword in this model. Continue with next model")
            continue
      posttrain(ExpID, runID, INPUT_name, OUTPUT_name)


load_test(experiment.experiment_id)
load_proof(experiment.experiment_id)

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
#             check = input("Type y if LOSS is acceptable \n Press Enter to continue...")"""
      
"""for n in range(2**4, ):
      for l in range(2**13):
            train(NNsize=int(n), 
                  total_epochs=5, 
                  length_item=int(l), 
                  OUTPUT_name=["lag 0"],# , "moving average", "lag 50"], #, "lag 100", "lag 250", "lag 500"]) # OUTPUT_name=["lag 50"], 
                  Arch = "Conv-AE-LSTM-P")"""
                  
#################################################################################################
# CVP Diagramms of feature prediction
""" - no SNR
    - Attention
    - no Weighted Pseudo-Loss
    - verzweigte Architektur
    - 0,5 Kernel
    - AdamW
    - Channel 2
    - Weighted SymbolC
"""
experiment = mlflow.set_experiment("Diagrams of feature prediction - CVP") 
print("Experiment_id: {}".format(experiment.experiment_id))

Out_list_ = [{'Tacho': ["lag 0"]},
             {'symbolsC': ["lag 0"]},
             {'Tacho': ["lag 0"], 'symbolsC': ["lag 0"]}
            ]

for Out in Out_list_:
            print(Out)
            train(
            total_epochs=5, 
            length_item= 300,
            OUTPUT_name = Out,
            Arch = "Conv_Att_E_final",
            dataset="CVP",
            kernel_size=0.5,
            weight_decay=True,
            weight_SymbolC=True)

load_test(experiment.experiment_id)
# load_proof(experiment.experiment_id)
