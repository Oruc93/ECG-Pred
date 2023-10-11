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
from scipy import stats

# Load model of current experiment
def load_proof(ExpID, total_epochs=250, 
          INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, # , "symbols":["lag 0"], "moving average: ["0.25"]
          NNsize=int(2 ** 4), length_item=int(2**9), Arch="Conv-AE-LSTM-P", dataset="SYNECG", FW_60=False):
    # Prepare dataset
    # DGl.preprocess(400, INPUT_name, {'SNR': ["lag 0"], 'Tacho': ["lag 0"], 'symbolsC': ["lag 0"], 'Shannon': ["lag 0"], 'Polvar10': ["lag 0"], 'forbword': ["lag 0"]}) # prepares data for generator. Only use if new dataset is used
    if not os.path.exists("plots_mape/" + str(ExpID)):
        # if directory is not present then create it.
        os.makedirs("plots_mape/" + str(ExpID))
    if not os.path.exists("plots_ae/" + str(ExpID)):
        # if directory is not present then create it.
        os.makedirs("plots_ae/" + str(ExpID))
    if not os.path.exists("hist_mape/" + str(ExpID)):
        # if directory is not present then create it.
        os.makedirs("hist_mape/" + str(ExpID))
    if not os.path.exists("hist_ae/" + str(ExpID)):
        # if directory is not present then create it.
        os.makedirs("hist_ae/" + str(ExpID))
    if not os.path.exists("plots_PvsT/" + str(ExpID)):
        # if directory is not present then create it.
        os.makedirs("plots_PvsT/" + str(ExpID))
    if not os.path.exists("heatmap/" + str(ExpID)):
        # if directory is not present then create it.
        os.makedirs("heatmap/" + str(ExpID))
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
                "task_loss_MAE": tl.task_loss_MAE,
                "symbols_loss_uniform": tl.symbols_loss_uniform,
                "symbols_loss_uniform_weighted": tl.symbols_loss_uniform_weighted} # Custom loss must be declared for loading model
    runs = mlflow.search_runs(ExpID, run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
    print(runs)
    # print(runs["run_id"])
    for n in range(len(runs)):
        runID = runs["run_id"][n]
        run_name = runs["tags.mlflow.runName"][n]
        if not(run_name in ["rare-ox-462",
                     "aged-sloth-522",
                     "honorable-gull-48",
                     "unleashed-toad-997",
                     "receptive-elk-55",
                     "luminous-ox-29"]): # nur erste Reihe der finalen Modelle bearbeiten 
            continue
        OUTPUT_name = eval(runs["params.Output features"][n])
        # print(runID)
        print("Evaluating model ", run_name, " on study data in Exp ", ExpID)
        # print(OUTPUT_name)
        out_types = DGl.output_type(OUTPUT_name)
        print(out_types)
        if not("forbword" in out_types): # check if forbword is in Output
            print("No forbword in this model. Continue with next model")
            continue
        with mlflow.start_run(runID):  # separate run for each NN size and configuration "cee8e987df7649a99c8b031941396e9a"
            samplerate = 256
            
            if os.path.exists("mlruns/"+ ExpID +"/"+ runID +"/artifacts/model/data/model"):
                model = tf.keras.models.load_model("mlruns/"+ ExpID +"/"+ runID +"/artifacts/model/data/model", custom_objects=custom_loss)
            else:
                print("No model found in Run. Continue with next Run")
                continue

            training_generator = DGl.DataGenerator(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=16, ToT="Proof")
            y_pred = model.predict(x=training_generator)

            print("Load proof chunks into variable...")
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
            de_novo_dic_test = {"RX02608":[], "RX05718":[], "RX06212":[], "RX10611":[], "RX10618":[], "RX12801":[], "RX12826":[], "RX14305":[], "RX14806":[], "RX35002":[]}
            control_dic_test = {"RX17101":[], "RX10603":[], "RX05701":[], "RX10638":[], "RX35005":[], "RX05715":[], "RX05711":[], "RX17105":[], "RX12813":[], "RX10612":[]}
            de_novo_list = []
            control_list = []
            de_novo_list_test = []
            control_list_test = []
            # reform forbword Output
            if len(y_test)>1:
                if FW_60:
                    y_t_U = (y_test[k][:]-0.1)*60
                    y_p_U = (y_pred[k][:]-0.1)*60
                else:
                    y_t_U = (y_test[k][:]-0.1)*40
                    y_p_U = (y_pred[k][:]-0.1)*40
            else:
                if FW_60:
                    y_t_U = (y_test[0]-0.1)*60
                    y_p_U = (y_pred-0.1)*60
                else:
                    y_t_U = (y_test[0]-0.1)*40
                    y_p_U = (y_pred-0.1)*40
            # Collect data by tying patient_ID to forbword
            for segment in range(len(patient_ID)):
                # check in which group forbword of segment is
                if patient_ID[segment] in list(de_novo_dic.keys()):
                    # de_novo_list.append(y_p_U[n])
                    if y_t_U[segment] > 0:
                        de_novo_dic_test[patient_ID[segment]].append(y_t_U[segment])
                        de_novo_dic[patient_ID[segment]].append(y_p_U[segment])
                        de_novo_list_test.append(y_t_U[segment])
                        de_novo_list.append(y_p_U[segment])
                else:
                    if y_t_U[segment] < 52.5:
                        control_dic_test[patient_ID[segment]].append(y_t_U[segment])
                        control_dic[patient_ID[segment]].append(y_p_U[segment])
                        control_list_test.append(y_t_U[segment])
                        control_list.append(y_p_U[segment])
            # Copy of lists to be filtered later
            de_novo_list_test_fil = de_novo_list_test
            de_novo_list_fil = de_novo_list
            control_list_test_fil = control_list_test
            control_list_fil = control_list
            # calculate mean forbword for each patient
            for key in list(de_novo_dic.keys()):
                mean = np.round(np.median(de_novo_dic_test[key]),0)
                # filter out ad outliers
                for fw in de_novo_dic_test[key]:
                    if np.abs((mean-fw)/mean) > 0.01:
                        del de_novo_dic[key][de_novo_dic_test[key].index(fw)]
                        de_novo_dic_test[key].remove(fw)
                        try:
                            del de_novo_list_fil[de_novo_dic_test[key].index(fw)]
                            de_novo_list_test_fil.remove(fw)
                        except:
                            None
                # de_novo_dic_test[key] = np.mean(de_novo_dic_test[key])
                de_novo_dic_test[key], counts = stats.mode(np.round(de_novo_dic_test[key],2))
                # de_novo_dic[key] = np.mean(de_novo_dic[key])
                de_novo_dic[key], counts = stats.mode(np.round(de_novo_dic[key],2))
            for key in list(control_dic.keys()):
                mean = np.round(np.median(control_dic_test[key]),0)
                for fw in control_dic_test[key]:
                    if np.abs((mean-fw)/mean) > 0.01:
                        del control_dic[key][control_dic_test[key].index(fw)]
                        # del y_p_U[y_t_U.index(fw)]
                        control_dic_test[key].remove(fw)
                        try:
                            del control_list_fil[control_dic_test[key].index(fw)]
                            control_list_fil.remove(fw)
                        except:
                            None
                        # y_t_U.remove(fw)
                # control_dic_test[key] = np.mean(control_dic_test[key])
                control_dic_test[key], counts = stats.mode(np.round(control_dic_test[key],2))
                # control_dic[key] = np.mean(control_dic[key])
                control_dic[key], counts = stats.mode(np.round(control_dic[key],2))
            print("de_novo_dic= ", de_novo_dic)
            print("control_dic= ", control_dic)
            print("de_novo_dic_test= ", de_novo_dic_test)
            print("control_dic_test= ", control_dic_test)

            U1, p = mannwhitneyu(list(de_novo_dic.values()), list(control_dic.values()))
            U2 = len(list(de_novo_dic.values())) * len(list(control_dic.values())) - U1
            U1_test, p = mannwhitneyu(list(de_novo_dic_test.values()), list(control_dic_test.values()))
            U2_test = len(list(de_novo_dic_test.values())) * len(list(control_dic_test.values())) - U1_test

            print("Statistics of M-U-Test ", U1, " + ", U2)
            print("Statistics of M-U-Test Truth ", U1_test, " + ", U2_test)
            if U1 <= 23 or U2 <= 23:
                M_U_check_test = True
                M_U_check = True # Nullhypothese ablehnen. beide Verteilungen sind unterschiedlich
                print("M-W-U-Test positiv")
            else:
                M_U_check_test = False
                M_U_check = False
                print("M-W-U-Test negative")
            check = True
            version = 0
            while check:
                try:
                    mlflow.log_param("Statistics of M-U-Test", U1)
                    mlflow.log_param("M-U-Test", M_U_check)
                    mlflow.log_param("Statistics of M-U-Test of Truth", U1_test)
                    mlflow.log_param("M-U-Test of Truth", M_U_check_test)
                    check = False
                    break
                except:
                    version += 1
                    try:
                        mlflow.log_param("Statistics of M-U-Test" + str(version), U1)
                        mlflow.log_param("M-U-Test" + str(version), M_U_check)
                        mlflow.log_param("Statistics of M-U-Test of Truth" + str(version), U1_test)
                        mlflow.log_param("M-U-Test of Truth" + str(version), M_U_check_test)
                        check=False
                        break
                    except:
                        print("Already Param saved. Saving version " + str(version))
                    
            
            # Plot MAPE depending on forbword of Truth
            # Calculate MAPE for each example
            print("Plotting Error behaviors...")
            
            Pseudo_Kombo = str(runs["params.Output features"][n].replace("['lag 0']",''))
            Pseudo_Kombo = Pseudo_Kombo.replace("{",'')
            Pseudo_Kombo = Pseudo_Kombo.replace("}",'')
            Pseudo_Kombo = Pseudo_Kombo.replace("'",'')
            Pseudo_Kombo = Pseudo_Kombo.replace(":",'')
            
            plt.figure(1)
            plt.title("forbword prediction vs Truth with combination\n" + Pseudo_Kombo)
            plt.plot(y_t_U, y_p_U, '.')
            plt.xlabel("true forbword values in a.u.")
            plt.ylabel("predicted forbword values in a.u.")
            bins = np.arange(10,66,2.5)
            plt.xticks(bins[::2])
            plt.yticks(bins[::2])
            plt.savefig("plots_PvsT/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + ".png")
            plt.close()
            
            mape = []
            ae = []
            for segment_ in range(len(y_t_U)):
                mape.append(abs((y_t_U[segment_] - y_p_U[segment_,0]) / y_t_U[segment_])*100)
                ae.append(abs(y_t_U[segment_] - y_p_U[segment_,0]))
            # mape = np.abs((y_t_U-y_p_U) / (y_t_U))*100
            # ae = np.abs(y_t_U-y_p_U)
            print(type(mape))
            print(len(mape))
            print(np.shape(y_p_U))
            print(np.shape(y_t_U))
            print("Calculated Error behaviors...")
            # exit()
            plt.figure(1)
            plt.title("MAPE of forbword prediction with combination\n" + Pseudo_Kombo)
            plt.plot(y_t_U, mape, '.')
            plt.xlabel("true forbword values in a.u.")
            plt.ylabel("MAPE of predicted forbword value in %")
            bins = np.arange(10,66,2.5)
            plt.xticks(bins[::2])
            plt.savefig("plots_mape/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + ".png")
            plt.close()
            print("Plotted MAPE...")
            
            plt.figure(1)
            plt.title("AE of forbword prediction with combination\n" + Pseudo_Kombo)
            plt.plot(y_t_U, ae, '.')
            plt.xlabel("true forbword values in a.u.")
            plt.ylabel("AE of predicted forbword value in a.u.")
            bins = np.arange(10,66,2.5)
            plt.xticks(bins[::2])
            plt.savefig("plots_ae/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + ".png")
            plt.close()
            print("Plotted AE...")
            
            plt.figure(1)
            plt.title("histogram of MAPE with combination\n" + Pseudo_Kombo)
            plt.hist(mape, bins=np.arange(0,200,2.5), density=True)
            plt.ylabel("density")
            plt.xlabel("MAPE of predicted forbword value in %")
            plt.savefig("hist_mape/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + ".png")
            plt.close()
            print("Plotted MAPE histo...")
            
            plt.figure(1)
            plt.title("histogram of AE with combination\n" + Pseudo_Kombo)
            plt.hist(ae,bins=30, density=True)
            plt.ylabel("density")
            plt.xlabel("AE of predicted forbword value in a.u.")
            plt.savefig("hist_ae/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + ".png")
            plt.close()
            print("Plotted AE histo...")
            
            plt.figure(1)
            plt.title("heatmap of prediction and Truth of FW with combination\n" + Pseudo_Kombo)
            plt.hist2d(y_t_U, y_p_U[:,0],bins=bins)
            plt.xticks(bins[::2])
            plt.yticks(bins[::2])
            plt.xlabel("true forbword values in a.u.")
            plt.ylabel("predicted forbword values in a.u.")
            plt.savefig("heatmap/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + ".png")
            plt.close()
            print("Plotted FW Heatmap...")
            
            # bins
            bins = np.arange(10,66,2.5)
            n_de_novo, bins_de_novo = np.histogram(de_novo_list, bins=bins)
            n_control, bins_control = np.histogram(control_list, bins=bins)
            n_de_novo_test, bins_de_novo_test = np.histogram(de_novo_list_test, bins=bins)
            n_control_test, bins_control_test = np.histogram(control_list_test, bins=bins)
            
            plt.figure(1)
            # plt.hist(n_de_novo, bins=bins+1.25, align="mid", rwidth=0.5)
            # plt.hist(n_control, bins=bins-1.25, align="mid", rwidth=0.5)
            plt.bar(bins_de_novo[1:], n_de_novo/sum(n_de_novo), align='edge', width=1)
            plt.bar(bins_control[1:], n_control/sum(n_control), align='edge', width=-1)
            plt.title("Distribution of predicted parameter forbword of segments in study")
            plt.xlabel("forbidden word [a.u]")
            plt.ylabel("density")
            plt.xticks(bins[::2])
            plt.legend(["de novo group", "control group"])
            plt.savefig("heatmap/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + "-Distribution-study-pred-forbword.png")
            plt.close()
            
            plt.figure(1)
            # plt.hist(n_de_novo, bins=bins+1.25, align="mid", rwidth=0.5)
            # plt.hist(n_control, bins=bins-1.25, align="mid", rwidth=0.5)
            plt.bar(bins_de_novo_test[1:], n_de_novo_test/sum(n_de_novo_test), align='edge', width=1)
            plt.bar(bins_control_test[1:], n_control_test/sum(n_control_test), align='edge', width=-1)
            plt.title("Distribution of real parameter forbword of segments in study")
            plt.xlabel("forbidden word [a.u]")
            plt.ylabel("density")
            plt.xticks(bins[::2])
            plt.legend(["de novo group", "control group"])
            plt.savefig("heatmap/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + "-Distribution-study-truth-forbword.png")
            plt.close()
            
            del n_de_novo, bins_de_novo, n_control, bins_control, n_de_novo_test, bins_de_novo_test, n_control_test, bins_control_test
            # Filtered Predictions and Truth FW
            # bins
            bins = np.arange(10,66,2.5)
            n_de_novo, bins_de_novo = np.histogram(de_novo_list_fil, bins=bins)
            n_control, bins_control = np.histogram(control_list_fil, bins=bins)
            n_de_novo_test, bins_de_novo_test = np.histogram(de_novo_list_test_fil, bins=bins)
            n_control_test, bins_control_test = np.histogram(control_list_test_fil, bins=bins)
            
            plt.figure(1)
            # plt.hist(n_de_novo, bins=bins+1.25, align="mid", rwidth=0.5)
            # plt.hist(n_control, bins=bins-1.25, align="mid", rwidth=0.5)
            plt.bar(bins_de_novo[1:], n_de_novo/sum(n_de_novo), align='edge', width=1)
            plt.bar(bins_control[1:], n_control/sum(n_control), align='edge', width=-1)
            plt.title("Distribution of predicted parameter forbword of segments in study")
            plt.xlabel("forbidden word [a.u]")
            plt.ylabel("density")
            plt.xticks(bins[::2])
            plt.legend(["de novo group", "control group"])
            plt.savefig("heatmap/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + "-Distribution-study-pred-forbword-fil.png")
            plt.close()
            
            plt.figure(1)
            # plt.hist(n_de_novo, bins=bins+1.25, align="mid", rwidth=0.5)
            # plt.hist(n_control, bins=bins-1.25, align="mid", rwidth=0.5)
            plt.bar(bins_de_novo_test[1:], n_de_novo_test/sum(n_de_novo_test), align='edge', width=1)
            plt.bar(bins_control_test[1:], n_control_test/sum(n_control_test), align='edge', width=-1)
            plt.title("Distribution of real parameter forbword of segments in study")
            plt.xlabel("forbidden word [a.u]")
            plt.ylabel("density")
            plt.xticks(bins[::2])
            plt.legend(["de novo group", "control group"])
            plt.savefig("heatmap/" + str(ExpID) + "/" + Pseudo_Kombo.replace(" ","") + "-Distribution-study-truth-forbword-fil.png")
            plt.close()
            
            """if not(isinstance(y_pred,list)): # check, ob y_pred list ist. Falls mehrere Outputs, dann ja
                y_pred = [y_pred]"""
            
            # Transform output of sparse categorical from 4D time series into 1D time series
            """for column in range(len(y_pred)): # loop over features
                if "classificationSymbols" in out_types[column]:
                    # sparse_pred = np.array([0,1,2,3]) * y_pred[column] # weighted sum of labels
                    # sparse_pred = np.sum(sparse_pred, axis=-1)
                    # y_pred[column] = sparse_pred
                    print(np.shape(y_pred[column]))
                    max_pred = np.argmax(y_pred[column], axis=-1)
                    print(np.shape(max_pred))
                    y_pred[column] = max_pred"""
            
            """example = np.random.randint(len(X_test[:,0]))"""
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
            
            """for k in range(len(y_pred)):
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
                        plt.close()"""
                        
def load_test(ExpID, total_epochs=250, 
          INPUT_name={"ECG":["lag 0"]}, OUTPUT_name={"ECG":["lag 0"]}, # , "symbols":["lag 0"], "moving average: ["0.25"]
          NNsize=int(2 ** 4), length_item=int(2**9), Arch="Conv-AE-LSTM-P", dataset="CVP", FW_60=False):
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
                "task_loss_MAE": tl.task_loss_MAE,
                "symbols_loss_uniform": tl.symbols_loss_uniform,
                "symbols_loss_uniform_weighted": tl.symbols_loss_uniform_weighted} # Custom loss must be declared for loading model
    runs = mlflow.search_runs(ExpID, run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
    print(runs)
    # print(runs["run_id"])
    for n in range(len(runs)):
        runID = runs["run_id"][n]
        run_name = runs["tags.mlflow.runName"][n]
        if run_name in ["rare-ox-462",
                     "aged-sloth-522",
                     "honorable-gull-48",
                     "unleashed-toad-997",
                     "receptive-elk-55",
                     "luminous-ox-29"]: # erste Reihe an Modellen ausschließen 
            continue
        else:
            try:
                index = list(runs["run_id"][:]).index(runs["tags.mlflow.parentRunId"][n])
                OUTPUT_name = eval(runs["params.Output features"][index])
            except:
                OUTPUT_name = eval(runs["params.Output features"][n])
        # print(runID)
        print("Evaluating model ", run_name, " on test data in Exp ", ExpID)
        # if run_name == "adaptable-owl-110":
        #     continue
        # print(OUTPUT_name)
        out_types = DGl.output_type(OUTPUT_name)
        print(out_types)
        """if not("forbword" in out_types): # check if forbword is in Output
            print("No forbword in this model. Continue with next model")
            continue"""
        with mlflow.start_run(runID):  # separate run for each NN size and configuration
            samplerate = 256
            if os.path.exists("mlruns/"+ ExpID +"/"+ runID +"/artifacts/model/data/model"):
                model = tf.keras.models.load_model("mlruns/"+ ExpID +"/"+ runID +"/artifacts/model/data/model", custom_objects=custom_loss)
            else:
                print("No model found in Run. Continue with next Run")
                continue
    
            # Evaluate trained model
            print("\nEvaluating model...")
            training_generator = DGl.DataGenerator(400, INPUT_name=INPUT_name, OUTPUT_name=OUTPUT_name, batch_size=16, ToT="Test")
            dic_eval = model.evaluate(x=training_generator, return_dict=True)
            print("Test results: ", dic_eval)
            version = 0
            for m in range(len(dic_eval)):
                # print(list(dic_eval)[m])
                if "forbword_output_mean_absolute_percentage_error" in list(dic_eval)[m] or "mean_absolute_percentage_error" == list(dic_eval)[m]:
                    check = True
                    while check:
                        try:
                            mlflow.log_param("MAPE forbword Test", dic_eval[list(dic_eval)[m]])
                            check = False
                        except:
                            version += 1
                            try:
                                mlflow.log_param("MAPE forbword Test" + str(version), dic_eval[list(dic_eval)[m]])
                                check = False
                            except:
                                print("Already Param saved. Saving version " + str(version))
            
            # Predict on test set and plot
            if not(dataset in ["CVP", "pretraining"]):
                y_pred = model.predict(X_test, batch_size=16)# int(np.shape(X_test)[0] / 3))
            else:
                y_pred = model.predict(x=training_generator)
                X_test, y_test, patient_ID = DGl.load_chunk_to_variable("Test", out_types)
            
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
                        if FW_60:
                            t = (y_test[k][example]-0.1)*60
                            p = (y_pred[k][example]-0.1)*60
                        else:
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
                    # plt.savefig(path_fig)  # saves plot
                    # mlflow.log_artifact(path_fig)  # links plot to MLFlow run
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
                            if FW_60:
                                t = (y_test[k][example]-0.1)*60
                                p = (y_pred[k][example]-0.1)*60
                            else:
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
                            if FW_60:
                                t = (y_test[k][example]-0.1)*60
                                p = (y_pred[k][example]-0.1)*60
                            else:
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
                        # if len(y_test) == 1: # if singular output, this fixes indexing
                        #     y_test[k] = [y_test[k]]
                        #     y_pred[k] = [y_pred[k]]
                        if out_types[k] in ["Shannon"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                            t = y_test[k][:]*4
                            p = y_pred[k][:]*4
                        elif out_types[k] in ["forbword"]: # rescaling prediction. Network is trained on scaled data. Scaled to interval [0,1]
                            if FW_60:
                                t = (y_test[k][example]-0.1)*60
                                p = (y_pred[k][example]-0.1)*60
                            else:
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
                        plt.legend(["y_Test", "Prediction"])
                        name = "./plots/data-Col-" + out_types[k] + "-all-examples"
                        plt.savefig(name)
                        z = np.random.randint(100000, 999999)
                        path_fig = "./Prediction-Image/" + str(z)+ "-data-" + out_types[k] + "-all-examples.png"
                        plt.savefig(path_fig)  # saves plot
                        mlflow.log_artifact(path_fig)  # links plot to MLFlow run
                        plt.close()
                        
def select_proof_segments():
    """
    Suche nach einer Zusammensetzung an Segmenten, um positives M-W-U-test zu erhalten
    Einteilung in kleinere Chunks nicht geholfen. Alle Chunks haben gleiche werte (mindestörße 400)
    Ausschluss der extremen ränder war Lösung
    
    Wir nehmen alle Segmente und filtern die extremen Ränder aus
    - in Kontrollgruppe
    """
    out_types = DGl.output_type({"forbword": ["lad 0"]})
    print(out_types)
    print("Load proof chunks into variable...")
    X_test, y_test, patient_ID = DGl.load_chunk_to_variable("Proof", out_types)
    print("Number of segments: ", len(patient_ID))
    print(np.shape(y_test))
    print(np.shape(patient_ID))
    y_chunks = []
    patient_chunks = []
    for n in range(int(np.shape(y_test)[1]/22000)):
        print(n)
        y_chunks.append(y_test[0][n:n+22000])
        patient_chunks.append(patient_ID[n:n+22000])
    print(np.shape(y_chunks))
    print(np.shape(patient_chunks))
    ###################################
    # Mann-Whitney-Houston-Test
    if not("forbword" in out_types):
        raise ValueError("forbword not calculated by NN. Please select another NN")
    k = out_types.index("forbword") # index of feature forbword
    de_novo_dic = {"RX02608":[], "RX05718":[], "RX06212":[], "RX10611":[], "RX10618":[], "RX12801":[], "RX12826":[], "RX14305":[], "RX14806":[], "RX35002":[]}
    control_dic = {"RX17101":[], "RX10603":[], "RX05701":[], "RX10638":[], "RX35005":[], "RX05715":[], "RX05711":[], "RX17105":[], "RX12813":[], "RX10612":[]}
    del y_test, patient_ID
    # loop over all chunks
    # calculate Test for each chunk and display result
    # use info to select appropiate chunks in load_chunk_to_variable
    print(len(y_chunks))
    Result = []
    for m in range(len(y_chunks)):
        print(n)
        print(np.shape([y_chunks[m]]))
        print(np.shape(patient_chunks[m]))
        de_novo_dic_test = {"RX02608":[], "RX05718":[], "RX06212":[], "RX10611":[], "RX10618":[], "RX12801":[], "RX12826":[], "RX14305":[], "RX14806":[], "RX35002":[]}
        control_dic_test = {"RX17101":[], "RX10603":[], "RX05701":[], "RX10638":[], "RX35005":[], "RX05715":[], "RX05711":[], "RX17105":[], "RX12813":[], "RX10612":[]}
        y_test = [y_chunks[m]]
        patient_ID = patient_chunks[m]
        if len(y_test)>1:
            y_t_U = (y_test[k][:]-0.1)*40
            # y_p_U = (y_pred[k][:]-0.1)*40
        else:
            y_t_U = (y_test[0]-0.1)*40
            # y_p_U = (y_pred-0.1)*40
        # Collect data by tying patient_ID to forbword
        for n in range(len(patient_ID)):
            # check in which group forbword of segment is
            if patient_ID[n] in list(de_novo_dic.keys()):
                # de_novo_list.append(y_p_U[n])
                if y_t_U[n] > 0:
                    de_novo_dic_test[patient_ID[n]].append(y_t_U[n])
                # de_novo_dic[patient_ID[n]].append(y_p_U[n])
            else:
                if y_t_U[n] < 52.5:# 52.5:
                    control_dic_test[patient_ID[n]].append(y_t_U[n])
                # control_dic[patient_ID[n]].append(y_p_U[n])
        # calculate mean forbword for each patient
        for key in list(de_novo_dic.keys()):
            # search for outliers in fw truth and exclude from analysis
            mean = np.round(np.median(de_novo_dic_test[key]),0)
            print(mean)
            # print(str(len(de_novo_dic_test[key])) + " segments for patient " + key)
            for fw in de_novo_dic_test[key]:
                if np.abs((fw-mean)/mean) > 0.01:
                    de_novo_dic_test[key].remove(fw)
            print(str(len(de_novo_dic_test[key])) + " segments left for patient " + key)
            # de_novo_dic_test[key] = np.mean(de_novo_dic_test[key])
            de_novo_dic_test[key], counts = stats.mode(np.round(de_novo_dic_test[key],0))
            print(de_novo_dic_test[key])
            # de_novo_dic[key] = np.mean(de_novo_dic[key])
        print("Calculate Control group")
        for key in list(control_dic.keys()):
            # search for outliers in fw truth and exclude from analysis
            mean = np.round(np.median(control_dic_test[key]),0)
            print(mean)
            # print(str(len(control_dic_test[key])) + " segments for patient " + key)
            for fw in control_dic_test[key]:
                if np.abs((fw-mean)/mean) > 0.01:
                    control_dic_test[key].remove(fw)
            print(str(len(control_dic_test[key])) + " segments left for patient " + key)
            # control_dic_test[key] = np.mean(control_dic_test[key])
            control_dic_test[key], counts = stats.mode(np.round(control_dic_test[key],0))
            print(control_dic_test[key])
            # control_dic[key] = np.mean(control_dic[key])
        # print(list(de_novo_dic.values()))
        # print(list(control_dic.values()))

        # U1, p = mannwhitneyu(list(de_novo_dic.values()), list(control_dic.values()))
        # U2 = len(list(de_novo_dic.values())) * len(list(control_dic.values())) - U1
        U1, p = mannwhitneyu(list(de_novo_dic_test.values()), list(control_dic_test.values()))
        U2 = len(list(de_novo_dic_test.values())) * len(list(control_dic_test.values())) - U1

        print("Statistics of M-U-Test ", U1, " + ", U2)
        if min(U1, U2) <= 23:
            M_U_check_test = True
            M_U_check = True # Nullhypothese ablehnen. beide Verteilungen sind unterschiedlich
        else:
            M_U_check_test = False
            M_U_check = False
        print(U1)
        print(M_U_check)
        Result.append([m, U1, M_U_check])
        print(Result)

"""experiment = mlflow.set_experiment("Experiment 6.2 - AdamW filling gaps - 80:20:dNC") # Conv-AE-LSTM-P good
print("Experiment_id: {}".format(experiment.experiment_id))
load_proof(experiment.experiment_id)"""
# select_proof_segments()