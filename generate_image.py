import mlflow
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pickle
import train_lib_final as tl
import random

INPUT_name={"ECG":["lag 0"]}
OUTPUT_name={"ECG":["lag 0"], "forbword":["lag 0"]}
data_list = list(OUTPUT_name.keys()) + list(INPUT_name.keys()) # list of mentioned features
data_list = tl.unique(data_list) # sorts out multiple occurences
print(data_list)
ecg_training, ecg_test, samplerate = tl.Icentia_memorize(50, 300, data_list)
exit()
print(np.shape(ecg_training))
for n in range(30):
    example = int(random.random()*np.shape(ecg_training)[0])
    plt.figure(10)
    time = np.arange(len(ecg_training[example,:2560]))/256
    plt.plot(time, ecg_training[example,:2560])
    plt.title("10 seconds ECG of patient in ReactDX study")
    plt.xlabel("time in s")
    plt.ylabel("amplitude in a.u.")
    # plt.xticks(bins[::2])
    # plt.legend(["de novo group", "control group"])
    plt.savefig("distributions/ECG-Icentia" + str(n) + ".png")