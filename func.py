import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import pandas as pd
from gwpy.spectrogram import Spectrogram


def get_data(filepath, ifo='L1'):
    f = h5py.File(filepath, 'r')
    df = pd.DataFrame(list(f['{}_q_data'.format(ifo)]))
    spec = Spectrogram(df, dt=0.01, df=0.05, f0=10)
    spec.times = np.linspace(-2, 2, 400)

    return spec


def above_threshold(dataframe, threshold):
    df = dataframe
    frac = np.sum(df[df>threshold].count())/np.sum(df.count())
    return frac


def qgram_sat(filepath, window, value, plot=False, savedir=False):
    t_win = window
    f1 = 100 
    f2 = 512
    f3 = 1024
    gpstime = int(filepath.split('_H1L1')[0][-10:])
    ifos = ['L1', 'H1']
    q_data = {}
    for ifo in ifos:
        try:
            q_data[ifo] = get_data(filepath, ifo)
        except:
            pass

    print(q_data.keys())
    

    if plot and q_data:
        ifo_q = q_data.keys()
        plot, axes = plt.subplots(len(q_data), 1, sharey=True, sharex=True, figsize=(12, 4*len(q_data)))
        for i, ax in zip(ifo_q, axes.flatten()):
            pcm = ax.imshow(q_data[i], vmin=0, vmax=25)
            ax.set_xlim(-t_win,t_win)
            ax.set_ylim(10, 1024)
            ax.set_ylabel('')
            ax.set_yscale('log')
            #ax.set_yticks([1.e+1, 1.e+2, 1.e+3])
            #ax.set_yticklabels([r'$10$', r'$100$', r'$10^3$'])
            ax.grid(alpha=0.3)
            # ax.text(-1.5,600,s='SNR: {}'.format(round(ifo_snrs[i],1)),
            #         color='lightcoral', fontsize='x-large', fontweight='bold')
            ax.xaxis.set_major_formatter(FormatStrFormatter(r'$%.1f$'))
            ax.plot([-100], label=i, visible=False)
            ax.legend(loc='upper left', handlelength=0, handletextpad=0)
        

        plot.text(0.52, 0.04, 'Time [s] from {}'.format(gpstime),
                  va='center', ha='center')
        # plot.text(0.52, 0.01, 'Event Id {}'.format(event_id),
        #       va='center', ha='center')
        plot.text(0.02, 0.53, r"$\mathrm{Frequency \ [Hz]}$",
                  va='center', ha='center', rotation='vertical')
        cbar = axes[0].colorbar(clim=(0, 25), location='top')
        cbar.set_label(r"$\mathrm{Normalized \ energy}$")


    if q_data['L1'] and q_data['H1']:
        dfL1, dfH1 = pd.DataFrame(q_data['L1']), pd.DataFrame(q_data['H1'])

    ### dividing rows and columns of the dataframe
    a, b = dfL1.shape[0]//2, dfL1.shape[1]//2 # center index
    len_one_sec =  int(dfL1.shape[0]/(2*t_win)) # int(1/0.01) 1/tres
    f1_index, f2_index, f3_index = int((f1 - 10)*1/0.05), int((f2 - 10)*1/0.05), int((f3 - 10)*1/0.05) #0.05 is fres
    row_ind = [len_one_sec//2, len_one_sec, len_one_sec*2]
    freq_ind = [0,f1_index, f2_index, f3_index]

    
    ### Zipping the row indices together
    row_inds = [(a-i, a+i) for i in row_ind]
    #### Zipping the freq indices together
    freq_inds = [(freq_ind[i], freq_ind[i+1]) for i in range(len(freq_ind)-1)]

    l1_above_thres = []
    h1_above_thres = []
    for j in freq_inds:
        for i in row_inds:
            l1_vals = above_threshold(dfL1.iloc[i[0]:i[1], j[0]:j[1]], value)
            l1_above_thres.append(100*l1_vals)
            h1_vals = above_threshold(dfH1.iloc[i[0]:i[1], j[0]:j[1]], value)
            h1_above_thres.append(100*h1_vals)

    dfL1tf = pd.DataFrame(np.reshape(l1_above_thres, (3, 3)), columns=['t1', 't2', 't3'],
                          index=['L1_f1', 'L1_f2', 'L1_f3'])
    # dfL1tf['gpstime'] = timel1
    dfH1tf = pd.DataFrame(np.reshape(h1_above_thres, (3, 3)), columns=['t1', 't2', 't3'],
                          index=['H1_f1', 'H1_f2', 'H1_f3'])
    # dfH1tf['gpstime'] = timeh1

    dfL1H1 = pd.concat([dfL1tf, dfH1tf], axis=0).round(1)
    dfL1H1['time'] = round(gpstime, 1)
    

    #filename = 'L1-H1-' + str(gpstime) + '.csv'
    filename = str(gpstime) + '_H1L1.csv'
    
    if savedir:
        output_file = os.path.join(savedir, filename)
        dfL1H1.to_csv(output_file)

    return dfL1H1, dfL1, dfH1
