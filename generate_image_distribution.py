import mlflow
import DataGen_lib as DGl
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pickle
import random

def load_y_training(ToT, out_types, bins):
    """reads out parameter from chunks
    calculates distribution chunkwise
    cumulates over all chunks

    Returns:

        y (list containing np.arrays): Output data. Contains feature which are to predicted by NN

    """
    with open('/mnt/scratchpad/dataOruc/data/current-set/CONFIG', 'rb') as fp:
        config = json.load(fp)
    with open('/mnt/scratchpad/dataOruc/data/current-set/OUTPUT_TYPE', 'rb') as file:
        output_type_chunks = json.load(file)
        # print(output_type_chunks)
        out_dic = {}
        for n in range(len(output_type_chunks)):
            out_dic[output_type_chunks[n]] = n
        print("data structure of chunks", out_dic)
    # length_item = int(config["segment_length"]) # length of ecg inputs in s
    # indexes = np.arange(int(config["segment_count"]))
    chunk_path_ = {"Training":'/mnt/scratchpad/dataOruc/data/current-set/', 
                     "Test":'/mnt/scratchpad/dataOruc/data/current-set/evaluation/',
                     "Proof":'/mnt/scratchpad/dataOruc/data/current-set/proof/'}
    chunk_path = chunk_path_[ToT]
    chunk_list = [f for f in os.listdir(chunk_path) if (os.path.isfile(os.path.join(chunk_path, f)) and not("patient_id" in f))]
    chunk_list = [int(f[2:-4]) for f in chunk_list if "npy" in f] # list of numbering of chunks
    chunk_list.sort()
    for chunk_ID in chunk_list:        
        with open(chunk_path + 'y-' + str(chunk_ID), 'rb') as fp:
            y_chunk = pickle.load(fp)
        y = y_chunk[out_dic[out_types[0]]]
        y = (y[0]-0.1)*40
        # bins = np.arange(10,61,5)
        print(chunk_ID)
        if chunk_ID == 0:
            n_y, bins_y = np.histogram(y, bins=bins)
        else:
            n_y_, bins_y = np.histogram(y, bins=bins)
            n_y += n_y_
    return n_y/sum(n_y), bins_y


# Distributions of parameter forbword in datasets
OUTPUT_name = {"forbword": ["lag 0"]}
out_types = DGl.output_type(OUTPUT_name)

# Study data
X_proof, y_proof, patient_ID = DGl.load_chunk_to_variable("Proof", out_types)
print(y_proof)
y_proof = (y_proof[0]-0.1)*40

print(np.shape(X_proof))
for n in range(30):
    example = int(random.random()*np.shape(X_proof)[0])
    plt.figure(10)
    time = np.arange(len(X_proof[example,:2560]))/256
    plt.plot(time, X_proof[example,:2560])
    plt.title("10 seconds ECG of patient in ReactDX study")
    plt.xlabel("time in s")
    plt.ylabel("amplitude in channel 2 in a.u.")
    # plt.xticks(bins[::2])
    # plt.legend(["de novo group", "control group"])
    plt.savefig("distributions/ECG-Channel2-" + str(n) + ".png")
    plt.close()
            

# Seperate segments into the study groups
de_novo_dic = {"RX02608":[], "RX05718":[], "RX06212":[], "RX10611":[], "RX10618":[], "RX12801":[], "RX12826":[], "RX14305":[], "RX14806":[], "RX35002":[]}
de_novo_list = []
control_list = []
for n in range(len(patient_ID)):
    # check in which group forbword of segment is
    if patient_ID[n] in list(de_novo_dic.keys()):
        de_novo_list.append(y_proof[n])
    else:
        if y_proof[n] < 52.5:
            control_list.append(y_proof[n])
# Simulate Filtering of study group
"""for key in list(de_novo_dic.keys()):
                mean = np.round(np.median(de_novo_dic_test[key]),0)
                # filter out ad outliers
                for fw in de_novo_dic_test[key]:
                    if np.abs((mean-fw)/mean) > 0.01:
                        del de_novo_dic[key][de_novo_dic_test[key].index(fw)]
                        de_novo_dic_test[key].remove(fw)
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
                        # y_t_U.remove(fw)
                # control_dic_test[key] = np.mean(control_dic_test[key])
                control_dic_test[key], counts = stats.mode(np.round(control_dic_test[key],2))
                # control_dic[key] = np.mean(control_dic[key])
                control_dic[key], counts = stats.mode(np.round(control_dic[key],2))"""


# bins
bins = np.arange(10,66,2.5)
n_de_novo, bins_de_novo = np.histogram(de_novo_list, bins=bins)
n_control, bins_control = np.histogram(control_list, bins=bins)

plt.figure(1)
# plt.hist(n_de_novo, bins=bins+1.25, align="mid", rwidth=0.5)
# plt.hist(n_control, bins=bins-1.25, align="mid", rwidth=0.5)
plt.bar(bins_de_novo[1:], n_de_novo/sum(n_de_novo), align='edge', width=1)
plt.bar(bins_control[1:], n_control/sum(n_control), align='edge', width=-1)
plt.title("Distribution of parameter forbword of segments in study")
plt.xlabel("forbidden word [a.u]")
plt.ylabel("density")
plt.xticks(bins[::2])
plt.legend(["de novo group", "control group"])
plt.savefig("distributions/Distribution-study-forbword.png")

# Test data
n_y, bins_y = load_y_training("Test", out_types, bins)
# X_test, y_test, patient_ID = DGl.load_chunk_to_variable("Test", out_types)
# print(y_test)
# y_test = (y_test[0]-0.1)*40

# n_training, bins_training = np.histogram(y_training, bins=bins)

plt.figure(2)
plt.bar(bins_y[1:], n_y, width=2)
# plt.hist(y_test, bins=bins)
# plt.hist(n_control, bins=bins-1.25, align="mid", rwidth=0.5)
# plt.bar(bins_de_novo[1:], n_de_novo, width=2)
plt.title("Distribution of parameter forbword of segments in test")
plt.xlabel("forbidden word [a.u]")
plt.ylabel("density")
plt.xticks(bins[::2])
# plt.legend(["de novo group", "control group"])
plt.savefig("distributions/Distribution-Testset-forbword.png")

# Training data
n_y, bins_y = load_y_training("Training", out_types, bins)

plt.figure(3)
plt.bar(bins_y[1:], n_y, width=2)
plt.title("Distribution of parameter forbword of segments in training")
plt.xlabel("forbidden word [a.u]")
plt.ylabel("density")
plt.xticks(bins[::2])
plt.savefig("distributions/Distribution-Trainingset-forbword.png")

###########################################################################################
# Generate ECG of channel 1 in RX
DGl.preprocess("Channel 1")
# Distributions of parameter forbword in datasets
OUTPUT_name = {"forbword": ["lag 0"]}
out_types = DGl.output_type(OUTPUT_name)

# Study data
X_1, y_1, patient_ID = DGl.load_chunk_to_variable("Channel 1", out_types)

print(np.shape(X_1))
for n in range(30):
    example = int(random.random()*np.shape(X_1)[0])
    plt.figure(10)
    time = np.arange(len(X_1[example,:2560]))/256
    plt.plot(time, X_1[example,:2560])
    plt.title("10 seconds ECG of patient in ReactDX study")
    plt.xlabel("time in s")
    plt.ylabel("amplitude in channel 1 in a.u.")
    # plt.xticks(bins[::2])
    # plt.legend(["de novo group", "control group"])
    plt.savefig("distributions/ECG-Channel1-" + str(n) + ".png")
    plt.close()