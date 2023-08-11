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
from scipy.stats import mannwhitneyu

# Load model of current experiment
def load(ExpID, total_epochs=250, 
          INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, # , "symbols":["lag 0"], "moving average: ["0.25"]
          NNsize=int(2 ** 4), length_item=int(2**9), Arch="Conv-AE-LSTM-P", dataset="SYNECG"):
    # Prepare dataset
    # DGl.preprocess(400, INPUT_name, {'SNR': ["lag 0"], 'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]}) # prepares data for generator. Only use if new dataset is used
    
    # Initiate dictionaries for LOSS and METRICS
    dic_loss = {"ECG":                  ["ECG_output", tl.pseudo_loss, 'MAE'],
                "Tacho":                ["Tacho_output", tl.pseudo_loss, 'MAE'],
                "symbolsC":             ["Symbols_output", tl.symbols_loss_uniform, 'sparse_categorical_accuracy'],
                "Shannon":              ["Shannon_output", tl.pseudo_loss, 'mean_absolute_percentage_error'], 
                "Polvar10":             ["Polvar10_output", tl.pseudo_loss, 'mean_absolute_percentage_error'], 
                "forbword":             ["forbword_output", tl.task_loss, 'mean_absolute_percentage_error'],
                "SNR":                  ["SNR_output", tl.pseudo_loss, 'MAE'],
                "words":                ["Words_output", tl.pseudo_loss, 'MAE'],
                "parameters":           ["parameter_output", tl.pseudo_loss, 'MAE'],
                "parametersTacho":      ["parameter_output", tl.pseudo_loss, 'MAE'], 
                "parametersSymbols":    ["parameter_output", tl.pseudo_loss, 'MAE'], 
                "parametersWords":      ["parameter_output", tl.pseudo_loss, 'MAE']
                }
    custom_loss = {"ECG_loss": tl.ECG_loss,
                "pseudo_loss": tl.pseudo_loss,
                "task_loss": tl.task_loss,
                "symbols_loss_uniform": tl.symbols_loss_uniform} # Custom loss must be declared for loading model
    runs = mlflow.search_runs(ExpID, run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
    print(runs)
    # print(runs["run_id"])
    for n in range(len(runs)):
        runID = runs["run_id"][n]
        run_name = runs["tags.mlflow.runName"][n]
        OUTPUT_name = eval(runs["params.Output features"][n])
        # print(runID)
        print("Evaluating model ", run_name, " on study data")
        # print(OUTPUT_name)
        out_types = DGl.output_type(OUTPUT_name)
        print(out_types)
        if not("forbword" in out_types): # check if forbword is in Output
            print("No forbword in this model. Continue with next model")
            continue
        with mlflow.start_run(runID):  # separate run for each NN size and configuration "cee8e987df7649a99c8b031941396e9a"
            
            samplerate = 256
            
            # with tf.keras.utils.custom_object_scope(dic_loss):
                # model = mlflow.keras.load_model("runs:/b7b93f47d96e47a6baaa6641db891495" + "/model")
            model = tf.keras.models.load_model("mlruns/"+ ExpID +"/"+ runID +"/artifacts/model/data/model", custom_objects=custom_loss)
            training_generator = DGl.DataGenerator(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=10, ToT="Proof")
            dic_eval = model.evaluate(x=training_generator, return_dict=True)
            y_pred = model.predict(x=training_generator)
            print("Load test chunks into variable...")
            X_test, y_test, patient_ID = DGl.load_chunk_to_variable("Proof", out_types)
            print("Number of segments: ", len(patient_ID))
            # data_list = list(OUTPUT_name.keys()) + list(INPUT_name.keys()) # list of mentioned features
            # data_list = tl.unique(data_list)
            # data_dic = DGl.feat_to_dic(data, data_list)
            # X, y_list, out_types = DGl.set_items(data_dic, INPUT_name, OUTPUT_name, 300*samplerate)
            
            ###################################
            # Mann-Whitney-Houston-Test
            if not("forbword" in out_types):
                raise ValueError("forbword not calculated by NN. Please select another NN")
            k = out_types.index("forbword") # index of feature forbword
            # de_novo_patient_list = ["RX02608", "RX05718", "RX06212", "RX10611", "RX10618", "RX12801", "RX12826", "RX14305", "RX14806", "RX35002"]
            # control_patient_list = ["RX17101", "RX10603", "RX05701", "RX10638", "RX35005", "RX05715", "RX05711", "RX17105", "RX12813", "RX10612"]
            de_novo_dic = {"RX02608":[], "RX05718":[], "RX06212":[], "RX10611":[], "RX10618":[], "RX12801":[], "RX12826":[], "RX14305":[], "RX14806":[], "RX35002":[]}
            control_dic = {"RX17101":[], "RX10603":[], "RX05701":[], "RX10638":[], "RX35005":[], "RX05715":[], "RX05711":[], "RX17105":[], "RX12813":[], "RX10612":[]}
            # de_novo_list = []
            # control_list = []
            if len(y_test)>1:
                y_t_U = (y_test[k][:]-0.1)*40
                y_p_U = (y_pred[k][:]-0.1)*40
            else:
                y_t_U = (y_test[0]-0.1)*40
                y_p_U = (y_pred[0]-0.1)*40
            # Collect data by tying patient_ID to forbword
            for n in range(len(patient_ID)):
                # check in which group forbword of segment is
                if patient_ID[n] in list(de_novo_dic.keys()):
                    # de_novo_list.append(y_p_U[n])
                    de_novo_dic[patient_ID[n]].append(y_p_U[n])
                else:
                    control_dic[patient_ID[n]].append(y_p_U[n])
            # calculate mean forbword for each patient
            for key in list(de_novo_dic.keys()):
                de_novo_dic[key] = np.mean(de_novo_dic[key])
            for key in list(control_dic.keys()):
                control_dic[key] = np.mean(control_dic[key])
            print(list(de_novo_dic.values()))
            print(list(control_dic.values()))
            U1, p = mannwhitneyu(list(de_novo_dic.values()), list(control_dic.values()))
            U2 = len(list(de_novo_dic.values())) * len(list(control_dic.values())) - U1
            print("Statistics of M-U-Test ", U1, " + ", U2)
            if min(U1, U2) <= 23:
                M_U_check = True # Nullhypothese ablehnen. beide Verteilungen sind unterschiedlich
            else:
                M_U_check = False
            mlflow.log_param("Statistics of M-U-Test", U1)
            mlflow.log_param("M-U-Test", M_U_check)
            
            
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
            
            example = np.random.randint(len(X_test[:,0]))
            # plt.figure(1)
            # plt.title("Full plot Truth and Pred of column 0")
            # plt.plot(list(range(len(y_test[0][0,:]))), y_test[0][example,:])
            # plt.plot(list(range(len(y_pred[0][0,:]))), y_pred[0][example,:])
            # if "ECG" in out_types[0]:
            #         plt.plot(list(range(len(X_test[0,:]))), X_test[example,:])
            # plt.legend(["y_test", "y_pred", "X_test"])
            # plt.savefig("Full-Plot col 0")
            # # while loop for saving image
            # k = int(0)
            # while k<100:
            #     k += 1
            #     z = np.random.randint(100000, 999999)
            #     path_fig = "./Prediction-Image/" + str(z) + ".png"
            #     if not os.path.isfile(path_fig):  # checks if file already exists
            #         plt.savefig(path_fig)  # saves plot
            #         mlflow.log_artifact(path_fig)  # links plot to MLFlow run
            #         break
            #     if k == 100:
            #         print("Could not find image name. Raise limit of rand.int above")
            # plt.close()
            
            for k in range(len(y_pred)):
                plt.figure(k)
                if out_types[k] in ["Shannon", "Polvar10", "forbword"]: # Check if column is non-linear parameter
                    print(k)
                    print(out_types[k])
                    print(example)
                    print(len(y_test[k]))
                    # if len(y_pred) == 1: # if singular output, this fixes indexing
                    #         y_test[k] = [y_test[k]]
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
                plt.legend(["y", "Prediction", "X"])
                name = concat("./ZoomPlot-Col-",str(k))
                plt.savefig(name)
                a = 0
                while a<100:
                    a += 1
                    z = np.random.randint(100000, 999999)
                    # path_fig = "./Prediction-Image/" + str(z) + ".png"
                    path_fig = "./Prediction-Image/" + str(z)+ "-Zoom-col" + str(a) + ".png"
                    if not os.path.isfile(path_fig):  # checks if file already exists
                        plt.savefig(path_fig)  # saves plot
                        mlflow.log_artifact(path_fig)  # links plot to MLFlow run
                        break
                    if a == 100:
                        print("Could not find image name. Raise limit of rand.int above")
                    plt.savefig(path_fig)  # saves plot
                    mlflow.log_artifact(path_fig)  # links plot to MLFlow run
                plt.close()
                
            # Plot multiple examples
            for l in range(10):
                example = np.random.randint(len(X_test[:,0]))
                for k in range(len(y_pred)):
                    # Full visuzilation
                    plt.figure(k)
                    if out_types[k] in ["Shannon", "Polvar10", "forbword"]: # Check if column is non-linear parameter
                        # if len(y_test) == 1: # if singular output, this fixes indexing
                        #     y_test[k] = [y_test[k]]
                        #     y_pred[k] = [y_pred[k]]
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
                    plt.legend(["y", "Prediction", "X"])
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
                    plt.legend(["y", "Prediction", "X"])
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
                        # if len(y_test) == 1: # if singular output, this fixes indexing
                        #     y_test[k] = [y_test[k]]
                        #     y_pred[k] = [y_pred[k]]
                        if out_types[k] in ["Shannon"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                            t = y_test[k][:]*4
                            p = y_pred[k][:]*4
                        elif out_types[k] in ["forbword"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                            t = (y_test[k][:]-0.1)*40
                            p = (y_pred[k][:]-0.1)*40
                        else:
                            t = y_test[k][:]-0.1
                            p = y_pred[k][:]-0.1
                        re = np.round(np.mean(abs(y_test[k][:]-y_pred[k][:])/abs(y_test[k][:])), decimals=2) # das hier nochmal durch testen
                        plt.figure(1)
                        plt.title("Parameter " + out_types[k] + " of all examples with rel. error " + str(re)) # rel. error einfügen
                        plt.plot(list(range(len(t))), t)
                        plt.plot(list(range(len(p))), p)
                        plt.legend(["y", "Prediction"])
                        name = "./plots/data-Col-" + out_types[k] + "-all-examples"
                        plt.savefig(name)
                        z = np.random.randint(100000, 999999)
                        path_fig = "./Prediction-Image/" + str(z)+ "-data-" + out_types[k] + "-all-examples.png"
                        plt.savefig(path_fig)  # saves plot
                        mlflow.log_artifact(path_fig)  # links plot to MLFlow run
                        plt.close()