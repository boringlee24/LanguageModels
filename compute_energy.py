import pandas as pd
import numpy as np
import os

def compute_energy(df):
    ts = pd.to_datetime(df['timestamp'])
    ts = ts - ts[0]
    ts = ts.dt.total_seconds().to_numpy()
    # Quadrature by trapezoidal rule
    deltas = ts[1:] - ts[0: -1]
    power = df[' power.draw [W]'].to_numpy()
    avg_powers = 0.5 * (power[1:] + power[0: -1])
    energy = deltas * avg_powers  # units of watts * seconds
    return np.sum(energy)

def get_energy(filename):
    df = pd.read_csv(filename)
    df0 = df.loc[df[' index'] == 0].reset_index(drop=True)
    df1 = df.loc[df[' index'] == 1].reset_index(drop=True)
    e0 = compute_energy(df0)
    e1 = compute_energy(df1)
    return e0, e1

def get_time(df):
    delta = pd.to_datetime(df.iloc[-1]) - pd.to_datetime(df.iloc[0])
    return delta.total_seconds()

def make_csv():
    basedir = "/home/gridsan/jpmcd/languagemodels"
    with open(os.path.join(basedir, "energy_nvidia_smi.csv"), "w") as f:
        jobs_dir = os.path.join(basedir, "bert-mlm/nvidia-smi")
        for job in sorted(os.listdir(jobs_dir)):
            print(job)
            job_path = os.path.join(jobs_dir, job)
            for csv in os.listdir(job_path):
                print(csv, end="\r")
                path = os.path.join(job_path, csv)
                e0, e1 = get_energy(path)
                line0 = ','.join([job, path, str(e0)]) + "\n"
                line1 = ','.join([job, path, str(e1)]) + "\n"
                f.write(line0)
                f.write(line1)


if __name__ == "__main__":
    if False:
        # CSV100 = '~/greenai/logs/43453210-1-hfmodel-bert-base-uncased-8-5-100W/43453210-c-7-12-1-nvidia-smi.csv'
        # CSV250 = '~/greenai/logs/43453601-1-hfmodel-bert-base-uncased-8-5-250W/43453601-c-7-12-2-nvidia-smi.csv'
        CSV100 = '~/greenai/logs/43531798-1-hfmodel-bert-base-uncased-8-5-100W/43531798-c-7-12-1-nvidia-smi.csv'
        CSV250 = '~/greenai/logs/43532443-1-hfmodel-bert-base-uncased-8-5-250W/43532443-c-7-12-2-nvidia-smi.csv'
        CSV100 = '/home/gridsan/JO30252/fastai_shared/for-joey/nvidia-smi/vgg-150W.csv'
        CSV250 = '/home/gridsan/JO30252/fastai_shared/for-joey/nvidia-smi/vgg-250W.csv'
        energy100 = get_energy(CSV100)
        energy250 = get_energy(CSV250)
        time100 = get_time(CSV100)
        time250 = get_time(CSV250)
        print('Energy used by 100W: {}'.format(energy100))
        print('Energy used by 250W: {}'.format(energy250))
        print('Relative Difference: {}'.format(1 - (energy100 / energy250)))
        print('Time for 100W: {} seconds'.format(time100))
        print('Time for 250W: {} seconds'.format(time250))
    make_csv()
