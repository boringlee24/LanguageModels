import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import json

scaling_ids = [
    '11094399', '11110709', '11094400', '11094401',
    '10328464', '10328474', '10973291', '10328462',
    '10328436', '10328472', '10973292', '10328434',
    '10328417', '10328470', '10328491', '10328412',
    '11400290', '11400289', '11400288', '11400267',
    '10328460', '10328466', '10328489', '10328458',
]

model_ids = []

inference_ids = ['11148966', '11148955', '11148944', '11147303']
batch16_ids = ['11400370', '11400368', '11400371', '11400366']

def load_jobs(filename):
    jobs = []
    with open(filename, 'r') as f:
        for l in f:
            jobs.append(eval(l))
    return jobs
    
def get_jobs():
    input_file = "energy_consumed.csv"
    input_file_smi = "energy_nvidia_smi.csv"
    job_stats = {
        'n_gpus': {},
        'batch_size': {},
        'epochs': {},
        'power': {},
        'energy': {},
        'time': {},
        'model': {},
        'dataset': {},
    }
    jobs = []
    with open(input_file, 'r') as f:
        for line in f:
            filename, energy = line.split(',')
            json_file = os.path.join(os.path.split(filename)[0], "config.json")
            with open(json_file, "r") as g:
                d = json.load(g)
                time = float(d["WALL_TIME"])
                model = d["TOKENIZER_NAME"]
                dataset = d["DATASET_CONFIG"]
            path = filename.split(os.sep)
            job_dir = path[6].split('-')
            job_id = job_dir[0]
            n_gpus = int(job_dir[1])
            batch_size = int(job_dir[2])
            epochs = float(job_dir[3])
            power = float(job_dir[4][:-1])
            energy = float(energy)
            job_stats['n_gpus'][job_id] = n_gpus
            job_stats['batch_size'][job_id] = batch_size
            job_stats['epochs'][job_id] = epochs
            job_stats['power'][job_id] = power
            # job_stats['energy'][job_id] = job_stats['energy'].get(job_id, 0.) + energy
            job_stats['time'][job_id] = time
            job_stats['model'][job_id] = model
            job_stats['dataset'][job_id] = dataset
    with open(input_file_smi, 'r') as f:
        for line in f:
            job_id, filename, energy = line.split(',')
            energy = float(energy)
            job_stats['energy'][job_id] = job_stats['energy'].get(job_id, 0.) + energy
    for job in job_stats['energy'].keys():
        # print(f"{job}: {job_stats['n_gpus'][job]} {job_stats['power'][job]} {job_stats['energy'][job] / (1000. * 3600.)}")
        # jobs.append([job, job_stats['n_gpus'][job], job_stats['power'][job], job_stats['epochs'][job], job_stats['energy'][job] / (1000. * 3600.), job_stats['time'][job] / 3600.])  # time measured in hours
        jobs.append([job, job_stats['n_gpus'][job], job_stats['power'][job], job_stats['batch_size'][job], job_stats['epochs'][job], job_stats['energy'][job] / (1000. * 3600.), job_stats['time'][job], job_stats['model'][job], job_stats['dataset'][job]])  # time measured in seconds
    jobs = [j for j in jobs if j[3] > 1.]
    # jobs.append(['xxxxxxxx', 392, 200.0, 40.0, 0., 0., 'None'])
    jobs.sort(key=lambda x: (x[1], x[4], x[3], x[7], x[2]))
    return jobs


def filter_jobs(jobs, n_gpus=None, epochs=None, batch_size=None, model=None, dataset=None, power=None):
    f1 = lambda x: n_gpus is None or x[1] == n_gpus
    f2 = lambda x: epochs is None or x[4] == epochs
    f3 = lambda x: batch_size is None or x[3] == batch_size
    f4 = lambda x: model is None or x[7] == model
    f5 = lambda x: dataset is None or x[8] == dataset
    f6 = lambda x: power is None or x[2] == power
    for f in [f1, f2, f3, f4, f5, f6]:
        jobs = filter(f, jobs)
    # fprod = lambda x: all([f(x) for f in [f1, f2, f3, f4, f5]])
    return list(jobs)


def plot_jobs(batches):
    fig, ax = plt.subplots()
    for i, batch in enumerate(batches):
        x = np.arange(len(batch))
        heights = [b[4] for b in batch]
        colors = ['C0', 'C1', 'C2', 'C3']
    #    ax.bar(i + x * .2 + .2, heights, .2, label=labels, color=colors)
    for i in range(4):
        x = np.arange(len(batches))
        heights = [batch[i][4] / batch[3][4] for batch in batches]
        ax.bar(x + i * .2 + .2, heights, .2, label=str(batches[0][1]))
    labels = [str(batch[0][1])+'-'+str(batch[0][3]) for batch in batches]
    ax.set_xticklabels([
        '64 - 4 epochs',
        '64 - 10 epochs',
        '128 - 15 epochs',
        '256 - 25 epochs',
        '392 - 40 epochs',
        '424 - 40 epochs'
        ],
        rotation=15
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.legend(['100W', '150W', '200W', '250W'], loc='lower left')
    filename = 'plot_power.png'
    # fig.savefig(filename, bbox_inches='tight')
    fig.savefig(filename)
    fig.show()

def plot_separate(batches):
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(10, 8)
    ax0 = fig.add_subplot(111, frameon=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Power Level (W)')
    ax0.set_ylabel('Energy Consumption (kWh)')
    ax00 = ax0.twinx()
    ax00.xaxis.set_visible(False)
    ax00.spines['right'].set_visible(False)
    ax00.spines['bottom'].set_visible(False)
    ax00.spines['top'].set_visible(False)
    ax00.spines['left'].set_visible(False)
    ax00.set_yticks([])
    ax00.set_ylabel('Time (hours)')
    ax0.xaxis.labelpad = 25
    ax0.yaxis.labelpad = 45
    ax00.yaxis.labelpad = 45
    for i, j in [(i, j) for i in range(3) for j in range(2)]:
        n = i * 2 + j
        batch = [b for b in batches[n] if b[4] != 0.]
        ax = axs[i][j]
        ax2 = ax.twinx()
        x = [1, 2, 3, 4] if len(batch) == 4 else [1, 2, 4]
        ticks = np.arange(1, 5)
        labels = [100, 150, 200, 250]
        watts = [b[4] for b in batch]
        time = [b[5] / 3600. for b in batch]
        annotation = "{:.1f} % less power\n{:.1f} % more time".format(100 * (1 - watts[1] / watts[-1]), 100 * (time[1] / time[-1] - 1))
        ax.annotate(annotation, (.25, .75), xycoords='axes fraction')
        l1, = ax.plot(x, watts, label='Power (W)', color='C0')
        l2, = ax2.plot(x, time, label='Time (s)', color='C1')
        title = "{} Nodes, {} Epochs".format(batch[0][1], int(batch[0][3]))
        ax.set_title(title)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(left=0.5)
        ax.set_xlim(right=4.5)
        # ax.set_ylim(bottom=0)
        # ax2.set_ylim(bottom=0)
    lgd = fig.legend([l1, l2],
        ['Power', 'Time'],
        borderaxespad=0.,
        bbox_to_anchor=(1., .95),
        loc="upper right"
    )
    lines = [l1, l2]
    labs = ['Power', 'Time']
    # fig.legend(lines, labs, loc='upper right', bbox_to_anchor=(.5, 1.025), borderaxespad=0, frameon=False)
    filename = 'plot_power_time.png'
    fig.tight_layout()
    fig.savefig(filename,
        # bbox_extra_artists=[lgd],
        # bbox_inches='tight'
    )
    fig.show()

def plot_one(batch, filename):
    # fig, ax = plt.subplots()
    fig, ax2 = plt.subplots()
    fig.set_size_inches(8, 3)
    # ax2 = ax.twinx()
    ax = ax2.twinx()
    ax2.set_xlabel('Max Power Level (W)', fontsize=10)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12, color='C1')
    ax2.set_ylabel('Training Time (hours)', fontsize=12, color='C0')
    ax2.yaxis.labelpad = 10
    batch = [b for b in batch if b[4] != 0.]
    x = [1, 2, 3, 4] if len(batch) == 4 else [1, 2, 4]
    ticks = np.arange(1, 5)
    watts = [b[4] for b in batch]
    time = [b[5] / 3600. for b in batch]
    l1, = ax.plot(x, watts, label='Power (W)', color='C1', linewidth=3)
    l2, = ax2.plot(x, time, label='Time (s)', color='C0', linewidth=3)
    topy = max(watts) * 1.1
    topy2 = max(time) * 1.1
    miny = min(watts) * .9
    miny2 = min(time) * .9
    title = "{} GPUs, {} Epochs".format(batch[0][1], int(batch[0][3]))
    # ax.set_title(title)
    ax.set_xticks(ticks)
    labels = [100, 150, 200, 250]
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlim(left=0.5)
    ax.set_xlim(right=4.5)
    # ax.set_ylim(bottom=12, top=topy)
    # ax2.set_ylim(bottom=0.5, top=topy2)
    ax.set_ylim(bottom=miny, top=topy)
    ax2.set_ylim(bottom=miny2, top=topy2)
    # ax.legend([l1, l2], ['Energy', 'Time'], loc='lower right')
    fig.tight_layout()
    # filename = "plot_{}_{}.png".format(batch[0][1], int(batch[0][3]))
    fig.savefig(filename,
        # bbox_extra_artists=[lgd],
        # bbox_inches='tight'
    )
    fig.show()
    

def plot_bars_side(batch, filename=None, subset=None):
    if subset is None:
        subset = batch[:-1]
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 3)
    time = [100 * (b[6] / batch[3][6]) for b in subset]
    energy = [100 * (b[5] / batch[3][5]) for b in subset]
    x = np.arange(1, 2 * len(time) + 1, 2)
    l1 = ax.barh(x - .4, time, .8, label='Time', color='C0', linewidth=3)
    l2 = ax.barh(x + .4, energy, .8, label='Energy', color='C1', linewidth=3)
    ax.axvline(100, color='k', linestyle='--')
    ax.invert_yaxis()
    topy = max(time) * 1.1
    miny = min(time) * .9
    if max(time) < 130:
        ax.set_xlim(right=130)
    if min(energy) > 60:
        ax.set_xlim(left=60)
    for w, z in zip(time, x - .4):
        ax.annotate('{:.1f}'.format(w), (ax.get_xlim()[0] + 2, z + .1), color='white', fontweight='bold')
    for w, z in zip(energy, x + .4):
        ax.annotate('{:.1f}'.format(w), (ax.get_xlim()[0] + 2, z + .1), color='white', fontweight='bold')
    ax.set_xlabel("Relative Performance (%)", fontsize=12) #, color='k', fontweight='bold')
    ax.set_ylabel("Power-cap Maximum (W)", fontsize=12)
    ax.legend(loc='lower right')
    ax.set_yticks(x)
    labels = [int(b[2]) for b in subset]
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    if filename is None:
        filename = "plot_bars_{}_{}.png".format(batch[0][1], int(batch[0][3]))
    fig.savefig(filename)


def plot_bars(batch, filename=None, title=None, subset=None):
    if subset is None:
        subset = batch[:-1]
    fig, ax = plt.subplots()
    fig.set_size_inches(2.5, 4)
    # ax.set_xlabel('Max Power Level (W)', fontsize=10)
    # ax.set_ylabel('Training Time (hours)', fontsize=12, color='C0')
    ax.set_ylabel("Relative Performance (%)", fontsize=12, color='k', fontweight='bold')
    # batch = [b for b in batch if b[4] != 0.]
    # energy = [b[5] for b in batch]
    # time = [b[6] / 3600. for b in batch]
    time = [100 * (b[6] / batch[3][6]) for b in subset]
    energy = [100 * (b[5] / batch[3][5]) for b in subset]
    # x = np.arange(1, 9, 2)
    x = np.arange(1, 2 * len(time) + 1, 2)
    l1 = ax.bar(x - .3, time, .6, label='Time', color='C0', linewidth=3)
    l2 = ax.bar(x + .3, energy, .6, label='Energy', color='C1', linewidth=3)
    ax.axhline(100, color='k', linestyle='--')
    topy = max(time) * 1.1
    miny = min(time) * .9
    if max(time) < 130:
        ax.set_ylim(top=130)
    if min(energy) > 60:
        ax.set_ylim(bottom=60)
    # ax2 = ax.twinx()
    # ax2.set_ylabel('Energy Consumption (kWh)', fontsize=12, color='C1')
    # ax2.yaxis.labelpad = 10
    # l2 = ax2.bar(x + .4, energy, .8, label='Power (W)', color='C1', linewidth=3)
    # topy2 = max(energy) * 1.1
    # miny2 = min(energy) * .9
    # ax2.set_ylim(bottom=miny2, top=topy2)
    # ax2.tick_params(axis='both', which='major', labelsize=10)
    if title is None:
        title = "Power-capping {} GPUs, BERT training on small dataset".format(batch[0][1], int(batch[0][3]))
    ax.set_title(title)
    ax.set_xticks(x)
    # labels = [100, 150, 200, 250]
    labels = [int(b[2]) for b in subset]
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    if filename is None:
        filename = "plotbars_{}_{}.png".format(batch[0][1], int(batch[0][3]))
    fig.savefig(filename,
        # bbox_extra_artists=[lgd],
        # bbox_inches='tight'
    )
    fig.show()

def plot_bars_all(batches, filename):
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 2.5)
    ax.set_ylabel("Relative Performance (%) ", fontsize=10,) # color='k', fontweight='bold')
    ax.set_xlabel("# of GPUs", fontsize=10,) # color='k', fontweight='bold')
    # ax.yaxis.labelpad = 10
    time100 = [100 * (b[0][6] / b[3][6]) for b in batches]
    time150 = [100 * (b[1][6] / b[3][6]) for b in batches]
    time200 = [100 * (b[2][6] / b[3][6]) for b in batches]
    energy100 = [100 * (b[0][5] / b[3][5]) for b in batches]
    energy150 = [100 * (b[1][5] / b[3][5]) for b in batches]
    energy200 = [100 * (b[2][5] / b[3][5]) for b in batches]
    time_avg100 = np.mean(time100)
    energy_avg100 = np.mean(energy100)
    time_avg150 = np.mean(time150)
    energy_avg150 = np.mean(energy150)
    time_avg200 = np.mean(time200)
    energy_avg200 = np.mean(energy200)
    print("Scaling - Time Avg (150W): ", time_avg150)
    print("Scaling - Energy Avg (150W): ", energy_avg150)
    labels = ["%s" % (b[0][1]) for b in batches]
    # nepochs = [b[0][3] for b in batches]
    x = np.arange(1, len(batches) * 2, 2)
    l0t = ax.bar(x - .75, time100, .3, color='C0', alpha=0.4, label="Time 100W")
    l0e = ax.bar(x - .45, energy100, .3, color='C1', alpha=0.4, label="Energy 100W")
    l1t = ax.bar(x - .15, time150, .3, color='C0', label="Time 150W")
    l1e = ax.bar(x + .15, energy150, .3, color='C1', label="Energy 150W")
    l2t = ax.bar(x + .45, time200, .3, color='navy', label="Time 200W")
    l2e = ax.bar(x + .75, energy200, .3, color='xkcd:dark orange', label="Energy 200W")
    ax.axhline(100, color='k', linestyle='--')
    # ax.axhline(time_avg100, color='C1', linestyle=':')
    # ax.axhline(energy_avg100, color='C1', alpha=0.5, linestyle=':')
    lat = ax.axhline(time_avg150, color='C0', linestyle='--', label='Avg Time 150W')
    lae = ax.axhline(energy_avg150, color='C1', linestyle='--', label='Avg Energy 150W')
    # ax.axhline(time_avg200, color='C2', linestyle=':')
    # ax.axhline(energy_avg200, color='C2', alpha=0.5, linestyle=':')
    # ax.annotate("250W Normal performance", (3, 100 + 1), color='k', fontsize=10, fontweight='bold')
    # ax.annotate("Avg Time Increase for 150W (%{:.0f})".format(time_avg - 100), (3.1, time_avg + 2), color='C0', fontsize=10, fontweight='bold')
    # ax.annotate("Avg Energy Decrease for 150W (%{:.0f})".format(100 - energy_avg), (4., energy_avg + 2), color='C1', fontsize=10, fontweight='bold')
    # topy = max(time) * 1.1
    # miny = min(time) * .9
    # topy2 = max(watts) * 1.1
    # miny2 = min(watts) * .9
    # ax.set_ylim(bottom=60, top=120)
    ax.set_ylim(bottom=60)
    ax.set_xlim(left=0, right=12)
    # title = "Performance of 150W compared to normal 250W, training BERT"
    # ax.set_title(title)
    ax.set_xticks(x)
    # ax.set_xticklabels(labels, fontweight='bold')
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    # ax.legend([l1, l2], ['Relative Time', 'Relative Energy'], loc='upper left')
    legend_handles = [l0t, l0e, l1t, l1e, l2t, l2e, lat, lae]
    legend_labels = ["Time 100W", "Energy 100W", "Time 150W", "Energy 150W", "Time 200W", "Energy 200W", "Avg Time 150W", "Avg Energy 150W"]
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig.tight_layout()
    fig.savefig(filename,
        # bbox_extra_artists=[lgd],
        # bbox_inches='tight'
    )
    fig.show()


def plot_bars_best(batches, filename):
    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 4)
    # ax2 = ax.twinx()
    ax2 = ax
    # ax.set_xlabel('Number GPUs / Epochs', fontsize=10)
    # ax.set_ylabel('Relative Training Time (%)', fontsize=12, color='C0')
    # ax2.set_ylabel('Relative Energy Consumption (%)', fontsize=12, color='C1')
    ax.set_ylabel("Relative Performance (%)", fontsize=12, color='k', fontweight='bold')
    ax2.yaxis.labelpad = 10
    time = [100 * (b[1][6] / b[3][6]) for b in batches]
    energy = [100 * (b[1][5] / b[3][5]) for b in batches]
    time_avg = np.mean(time)
    energy_avg = np.mean(energy)
    print("Time:", time)
    print("Energy:", energy)
    labels = ["%s" % (b[0][1]) for b in batches]
    # nepochs = [b[0][3] for b in batches]
    x = np.arange(1, len(batches) * 2, 2)
    l1 = ax.bar(x - .4, time, .8, color='C0', label="Relative Time")
    l2 = ax2.bar(x + .4, energy, .8, color='C1', label="Relative Energy")
    ax.axhline(100, color='k', linestyle='--')
    ax.axhline(time_avg, color='C0', linestyle='--')
    ax.axhline(energy_avg, color='C1', linestyle='--')
    # ax.annotate("250W Normal performance", (3, 100 + 1), color='k', fontsize=10, fontweight='bold')
    # ax.annotate("Avg Time Increase for 150W (%{:.0f})".format(time_avg - 100), (3.1, time_avg + 2), color='C0', fontsize=10, fontweight='bold')
    # ax.annotate("Avg Energy Decrease for 150W (%{:.0f})".format(100 - energy_avg), (4., energy_avg + 2), color='C1', fontsize=10, fontweight='bold')
    # topy = max(time) * 1.1
    # miny = min(time) * .9
    # topy2 = max(watts) * 1.1
    # miny2 = min(watts) * .9
    ax.set_ylim(bottom=60, top=120)
    # ax2.set_ylim(bottom=60, top=120)
    # title = "Performance of 150W compared to normal 250W, training BERT"
    # ax.set_title(title)
    ax.set_xticks(x)
    # ax.set_xticklabels(labels, fontweight='bold')
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    # ax2.tick_params(axis='both', which='major', labelsize=10)
    # ax.legend([l1, l2], ['Relative Time', 'Relative Energy'], loc='upper left')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(filename,
        # bbox_extra_artists=[lgd],
        # bbox_inches='tight'
    )
    fig.show()


def plot_hardware(filename):
    jobs = {}
    with open('nlp_pwr_cap.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # if row[0] == 'V100':
            #     continue
            if row[0] not in jobs:
                jobs[row[0]] = []
            jobs[row[0]].append([float(row[1]), float(row[2]), float(row[4])])
    fig, axs = plt.subplots(2, 2)
    ax = fig.add_subplot(111, frameon=False)
    # fig.set_size_inches(8, 3.5)
    fig.set_size_inches(4, 4)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # ax.set_xlabel("")
    ax.set_ylabel("Relative Performance (%)", labelpad=10.)
    for i, gpu in enumerate(jobs):
        s = i // 2
        t = i % 2
        time = [100 * (j[1] / jobs[gpu][2][1]) for j in jobs[gpu]]
        energy = [100 * (j[2] / jobs[gpu][2][2]) for j in jobs[gpu]]
        x = np.arange(1, 2 * len(time) + 1, 2)
        # l1 = axs[i].bar(x - .3, time, .6, label='Time', color='C0', linewidth=3)
        # l2 = axs[i].bar(x + .3, energy, .6, label='Energy', color='C1', linewidth=3)
        # axs[i].axhline(100, color='k', linestyle='--')
        l1 = axs[s][t].bar(x - .3, time, .6, label='Time', color='C0', linewidth=3)
        l2 = axs[s][t].bar(x + .3, energy, .6, label='Energy', color='C1', linewidth=3)
        axs[s][t].axhline(100, color='k', linestyle='--')
        topy = max(time) * 1.1
        miny = min(time) * .9
        if max(time) < 130:
            # axs[i].set_ylim(top=130)
            axs[s][t].set_ylim(top=130)
        if min(energy) > 60:
            # axs[i].set_ylim(bottom=60)
            axs[s][t].set_ylim(bottom=60)
        # axs[i].set_xlabel(gpu)
        # axs[i].set_xticks(x)
        axs[s][t].set_xlabel(gpu)
        axs[s][t].set_xticks(x)
        # labels = [100, 150, 200, 250]
        labels = ["%sW" % int(j[0]) for j in jobs[gpu]]
        # axs[i].set_xticklabels(labels)
        # axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[s][t].set_xticklabels(labels)
        axs[s][t].tick_params(axis='both', which='major', labelsize=10)
    # axs[0].legend(loc='upper right', frameon=False)
    axs[0][0].legend(loc='upper right', frameon=False)
    ax.set_ylabel("Relative Performance (%)", fontsize=12) # , color='k', fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')


def plot_models(batches, filename):
    fig, axs = plt.subplots(1, 3)
    ax = fig.add_subplot(111, frameon=False)
    fig.set_size_inches(6, 2.75)
    # fig.set_size_inches(6, 3.25)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    models = ['BERT', 'DistilBERT', 'BigBird']
    time100 = [100 * (batch[0][6] / batch[3][6]) for batch in batches]
    time150 = [100 * (batch[1][6] / batch[3][6]) for batch in batches]
    time200 = [100 * (batch[2][6] / batch[3][6]) for batch in batches]
    energy100 = [100 * (batch[0][5] / batch[3][5]) for batch in batches]
    energy150 = [100 * (batch[1][5] / batch[3][5]) for batch in batches]
    energy200 = [100 * (batch[2][5] / batch[3][5]) for batch in batches]
    print("Models - Time 150:", time150)
    print("Models - Energy 150:", energy150)
    print("Models - Time Avg (150W): ", np.mean(time150))
    print("Models - Energy Avg (150W): ", np.mean(energy150))
    for i, batch in enumerate(batches):
        subset = batch[:-1]
        time = [100 * (b[6] / batch[3][6]) for b in subset]
        energy = [100 * (b[5] / batch[3][5]) for b in subset]
        x = np.arange(1, 2 * len(time) + 1, 2)
        l1 = axs[i].bar(x - .3, time, .6, label='Time', color='C0', linewidth=3)
        l2 = axs[i].bar(x + .3, energy, .6, label='Energy', color='C1', linewidth=3)
        axs[i].axhline(100, color='k', linestyle='--')
        topy = max(time) * 1.1
        miny = min(time) * .9
        if max(time) < 130:
            axs[i].set_ylim(top=130)
        if min(energy) > 50:
            axs[i].set_ylim(bottom=50)
        if min(energy) > 60:
            axs[i].set_ylim(bottom=60)
        axs[i].set_xlabel(models[i])
        axs[i].set_xticks(x)
        labels = ["%sW" % int(b[2]) for b in subset]
        axs[i].set_xticklabels(labels)
        axs[i].tick_params(axis='both', which='major', labelsize=10)
    axs[0].set_yticks(range(60, 141, 10))
    axs[1].set_yticks(range(60, 131, 10))
    axs[2].set_yticks(range(50, 151, 10))
    axs[0].legend(loc='upper right', frameon=False, fontsize=8)
    # ax.set_ylabel("Relative Performance (%)", fontsize=12) # , color='k', fontweight='bold')
    ax.set_ylabel("Relative Performance (%)", fontsize=10, labelpad=10.)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    

def print_means(batches):
    power_saved = [100 * (1 - batch[1][4] / batch[-1][4]) for batch in batches]
    time_diff = [100 * (batch[1][5] / batch[-1][5] - 1) for batch in batches]
    print(np.mean(power_saved), power_saved)
    print(np.mean(time_diff), time_diff)
    print(np.mean(power_saved[1:]))
    print(np.mean(time_diff[1:]))


def make_csv():
    with open("output.txt", "r") as f:
        lines = f.readlines()
    with open("output.csv", "w") as f:
        f.write("JobID,nGPUs,MaxPower,BatchSize,Epochs,Energy(kWh),Time(s),Model,Dataset"+"\n")
        for line in lines:
            l = eval(line)
            if l[0] == "11113573":
                continue
            out = ",".join([str(x) for x in l])
            f.write(out+"\n")


if __name__ == "__main__":
    if True:
        jobs = get_jobs()
        jobs = filter_jobs(jobs, dataset='wikitext-103-raw-v1')
        with open('output.txt', 'w') as f:
            for j in jobs:
                f.write(str(j) + '\n')
    else:
        jobs = load_jobs('output.txt')
        # jobs = load_jobs('output_dcgm.txt')
    make_csv()
    if False:
        print("JobID,nGPUs,MaxPower,BatchSize,Epochs,Energy(kWh),Time(s),Model,Dataset")
        for job in jobs:
            print("{},{},{},{},{},{},{},{},{}".format(*job))
    if False:
        # batches = [jobs[i: i + 4] for i in range(0, len(jobs), 4)]
        batches = [x for x in jobs if x[0] in scaling_ids]
        batches = [batches[i: i + 4] for i in range(0, len(batches), 4)]
        plot_bars_all(batches, "/home/gridsan/jpmcd/db/green/plot_bars_all.png")
        plot_hardware("/home/gridsan/jpmcd/db/green/plot_bars_hardware.png")
        batch = [
            filter_jobs(jobs, n_gpus=16, epochs=4., batch_size=8, model="bert-base-uncased"),
            filter_jobs(jobs, n_gpus=16, epochs=4., batch_size=8, model="distilbert-base-uncased"),
            filter_jobs(jobs, n_gpus=16, epochs=4., batch_size=8, model="google/bigbird-roberta-base"),
        ]
        plot_models(batch, "/home/gridsan/jpmcd/db/green/plot_bars_models.png")
        # plot_bars_best(batches, "/home/gridsan/jpmcd/db/green/plot_bars_best.png")
        # batch = [x for x in jobs if x[0] in inference_ids]
        # plot_bars_side(batch, "/home/gridsan/jpmcd/db/green/plot_bars_inference.png", batch[:3])
        # batch = [x for x in jobs if x[0] in batch16_ids]
        # plot_bars(batch, "/home/gridsan/jpmcd/db/green/plot_bars_batch16.png", "Batch Size = 16")
        # plot_one(batches[3])
        # plot_bars(batches[3])
