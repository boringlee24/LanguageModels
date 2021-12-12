import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

filename = "/home/gridsan/jpmcd/languagemodels/bert-mlm/nvidia-smi/10328464/nvidia-smi-time-series-d-12-13-2-10328464.csv"
basedir = "/home/gridsan/jpmcd/languagemodels/bert-mlm/nvidia-smi"
# filename = "nvidia-smi-time-series-d-12-13-2-10328464.csv"
# basedir = os.path.abspath('nvidia-smi')
color_floats = np.mod(np.arange(0, 25, .78), 1)
colors = plt.cm.rainbow(color_floats)
jobs = ['10328417', '10328491', '10328470', '10328412']
handles = []
conv_length = 20
conv = np.ones(conv_length)
for i, job in enumerate(jobs):
    print(job)
    jobdir = os.path.join(basedir, job)
    color = colors[i]
    for filename in os.listdir(jobdir):
        print(filename)
        filename = os.path.join(jobdir, filename)
        df = pd.read_csv(filename)
        df = df.rename(columns={c: c.strip() for c in df})
        td = pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'].loc[0])
        df['timedeltas'] = td.dt.total_seconds()
        gpu0 = df.loc[df['index'] == 0].reset_index(drop=True)
        gpu1 = df.loc[df['index'] == 1].reset_index(drop=True)
        gpu0_avg_temp = gpu0['temperature.gpu'].to_numpy()
        gpu1_avg_temp = gpu1['temperature.gpu'].to_numpy()
        l0 = len(gpu0_avg_temp)
        l1 = len(gpu1_avg_temp)
        gpu0_temp_avg = np.convolve(gpu0_avg_temp, conv)[:l0] / np.minimum(np.arange(1, l0 + 1), conv_length)
        gpu1_temp_avg = np.convolve(gpu1_avg_temp, conv)[:l1] / np.minimum(np.arange(1, l1 + 1), conv_length)
        # plt.plot(gpu0['timedeltas'].to_numpy()[::conv_length], gpu0_temp_avg[::conv_length], '-', c=color, alpha=.2)
        plt.plot(gpu0['timedeltas'].to_numpy(), gpu0_temp_avg, '-', c=color, alpha=.2)
        hdl, = plt.plot(gpu1['timedeltas'].to_numpy(), gpu1_temp_avg, '-', c=color, alpha=.2)
    handles.append(hdl)
plt.legend(handles, jobs)
plt.savefig('/home/gridsan/jpmcd/db/plot_temps.png')
