import numpy as np
import matplotlib.mlab as mlab
import mne
import matplotlib.pyplot as plt
import collections
import os
import datetime
import glob
import gc
from sklearn.decomposition import PCA, FastICA, NMF
import scipy.io
import scipy.signal
import itertools
from concurrent.futures import ProcessPoolExecutor
import h5py
import sys

from helper_functions import *



proj_data_dir = get_proj_data_dir()
RCS0X = get_RCS0X()

def get_sliding_windows(fs, movingwin, Nt):
    Nwin = int(np.round(fs * movingwin[0])) # number of samples in window
    Nstep = np.round(fs * movingwin[1]) # number of samples to step through
    start_inds = list(range(0, int(Nt - Nwin + 1), int(Nstep)))
    if start_inds[-1] + Nwin > Nt:
        start_inds = start_inds[0:-1]
    return Nwin, Nstep, start_inds

def load_hdf5(filepath):
    with h5py.File(filepath, 'r') as h5file:
        data = h5file['data'][:]
        sfreq = h5file.attrs['sfreq']
        ch_names = [ch.decode('utf-8') if isinstance(ch, bytes) else ch for ch in h5file.attrs['ch_names']]
        description = h5file.attrs['description']
        meas_date = h5file.attrs['meas_date']
        return data, sfreq, ch_names, description, meas_date

def compute_average_power_in_bands(psd, freqs):
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 70),
        'high_gamma': (70, 150)
    }

    avg_power = {}
    for band, (low_freq, high_freq) in bands.items():
        # Find the indices corresponding to the frequency band
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        # Calculate the average power in this band
        avg_power[band] = np.mean(psd[:, idx_band], axis=1)
        #avg_power[band] = np.mean(psd[idx_band])

    return avg_power

def process_window(data, sr, TW, L, times, winsize, overlap, window_num, save_fold, spec_save_fold, chunk_num):
    Nwin, Nstep, inds = get_sliding_windows(sr, (winsize, overlap), data.shape[1])
    freqs = two_sided_freqs(sr, Nwin)
    win = inds[window_num]
    endi = win + Nwin
    datachunk = data[:, win:endi]
    #print(data.shape)
    win_fft, psd, eigvalues = compute_fft(datachunk, sr, TW, L)


    power_spectra = get_power(win_fft)
    power_spectra = power_spectra.mean(1) #avg tapers

    #max freq = 200
    freq_idxz = np.where(freqs <= 500)[0]
    print(len(freq_idxz))

    fr_sel_power_spectra = power_spectra[:,freq_idxz]
    fr_sel_freqs = freqs[freq_idxz]
    avged_spec = []
    binned_freqs = np.arange(1, 500)

    for fr in binned_freqs:
        fr_idxz = np.where((fr_sel_freqs >= fr) & (fr_sel_freqs < fr + 1))[0]
        avged_spec.append(np.mean(fr_sel_power_spectra[:,fr_idxz], axis = 1))
    avged_spec = np.array(avged_spec).T

    spec_save_fn = spec_save_fold + "spec_{}_chunk{}_win{}".format(get_RCS0X(), chunk_num, window_num)
    np.save(spec_save_fn, avged_spec)

    #save freqs too
    freq_save_fn = spec_save_fold + "freqs_{}_chunk{}_win{}".format(get_RCS0X(), chunk_num, window_num)
    np.save(freq_save_fn, binned_freqs)

    # Compute average power in canonical frequency bands
    avg_power = compute_average_power_in_bands(power_spectra, freqs)

    # Save average power for each frequency band
    #print('saving to ' + save_fold + "avg_power_{}_chunk{}_win{}".format(get_RCS0X(), chunk_num, window_num))
    save_fn = save_fold + "avg_power_{}_chunk{}_win{}".format(get_RCS0X(), chunk_num, window_num)
    np.save(save_fn, avg_power)

    return None  # Return None as we are saving within the function

if False: #run alldata
    data_fold = get_h5_dir()
    save_fold = proj_data_dir + 'alldat_psd/'

    if not os.path.exists(save_fold):
        os.makedirs(save_fold)

    files = glob.glob(data_fold + '*.h5')
    files = [f for f in files if "ttls" in f]
    tr_nums = []

    for file in files:
        tr_nums.append(file.split('/')[-1].split('_')[1][5:])

    tr_nums = np.array(tr_nums).astype(int)
    tr_nums = np.sort(tr_nums)

    def process_chunk(num):
        print("Processing chunk " + str(num))
        fname = f'{get_RCS0X()}_chunk' + str(num) + '_eeg_raw.h5'
        data, sr, ch_names, description, meas_date = load_hdf5(data_fold + fname)
        print(ch_names)

        TW = 3  # example value
        times = np.arange(data.shape[1]) / sr  # example value
        winsize = 2  # example value
        overlap = 0.5  # example value

        process_window(data, sr, TW, L, times, winsize, overlap, 0, save_fold, spec_save_fold, num)

        # with ProcessPoolExecutor(max_workers=8) as executor:  # Limit the number of workers
        #     futures = [executor.submit(process_window, data, sr, TW, L, times, winsize, overlap, j, save_fold, num) for j in range(len(inds))]
        #     for future in futures:
        #         future.result()  # Ensuring the future completes
        #         gc.collect()  # Force garbage collection

else:
    data_fold = get_surv_h5_dir()
    save_fold = foldcheck(proj_data_dir + 'apr_2025_duringsurv_psd_tw_12/')
    spec_save_fold = foldcheck(proj_data_dir + 'apr_2025_duringsurv_specs_tw_12/')

    if not os.path.exists(save_fold):
        os.makedirs(save_fold)

    if not os.path.exists(spec_save_fold):
        os.makedirs(spec_save_fold)

    files = glob.glob(data_fold + '*.h5')
    files = [f for f in files if "ttls" in f]
    tr_nums = []

    for file in files:
        tr_nums.append(os.path.basename(file).split('_')[1])

    tr_nums = np.array(tr_nums).astype(int)
    tr_nums = np.sort(tr_nums)

    def process_chunk(num):
        print("Processing chunk " + str(num))
        fname = f"{RCS0X}_" + str((num)) + "_eeg_raw.h5"

        data, sr, ch_names, description, meas_date = load_hdf5(data_fold + fname)

        #if data is of length 29999, add a zero to the end
        if data.shape[1] == 29999:
            data = np.concatenate((data, np.zeros((data.shape[0], 1))), axis=1)

        TW = 20
        times = np.arange(data.shape[1]) / sr  # example value
        winsize = 5  # example value
        overlap = 5  # example value

        L, freq_res = get_mt_info(TW, winsize, 'auto', print_stuff=True)

        Nwin, Nstep, inds = get_sliding_windows(sr, (winsize, overlap), data.shape[1])

        #save ch names
        #process_window(data, sr, TW, L, times, winsize, overlap, 0, save_fold, spec_save_fold, num)

        with ProcessPoolExecutor(max_workers=16) as executor:  # Limit the number of workers
            futures = [executor.submit(process_window, data, sr, TW, L, times,
                                       winsize, overlap, j, save_fold, spec_save_fold, num) for j in range(len(inds))]
            for future in futures:
                future.result()  # Ensuring the future completes
                gc.collect()  # Force garbage collection


if __name__ == "__main__":
    for tr_num in tr_nums:
        print(tr_num)
        process_chunk(tr_num)
    #process_chunk(tr_nums[10])
